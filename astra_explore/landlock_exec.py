"""Launch a command under a Landlock filesystem sandbox (Linux LSM, ABI >= 3) and exec it.

Everything not listed in the spec is invisible to the child: no read, list, write or execute. Rules are inherited
by every descendant and cannot be lifted. Network is not restricted (not handled). Used to run `codex exec` so the
model's process cannot inspect the simulator, the evaluator or any experiment file outside its own input folder.

    python landlock_exec.py SPEC.json -- COMMAND [ARGS...]
SPEC: {"rules": [{"path": "/usr", "access": "ro"}, {"path": "/dev", "access": "rw"},
                 {"path": "/home/u/.codex", "access": "files_rw_nolist"}, ...],
       "env": {"HOME": "...", ...}}      # the child gets exactly this environment
"""
import ctypes
import json
import os
import stat
import sys

SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446
RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38
A = {n: 1 << i for i, n in enumerate(["EXECUTE", "WRITE_FILE", "READ_FILE", "READ_DIR", "REMOVE_DIR", "REMOVE_FILE",
                                       "MAKE_CHAR", "MAKE_DIR", "MAKE_REG", "MAKE_SOCK", "MAKE_FIFO", "MAKE_BLOCK",
                                       "MAKE_SYM", "REFER", "TRUNCATE", "IOCTL_DEV"])}
FILE_OK = A["EXECUTE"] | A["WRITE_FILE"] | A["READ_FILE"] | A["TRUNCATE"] | A["IOCTL_DEV"]
SETS = {"ro": A["EXECUTE"] | A["READ_FILE"] | A["READ_DIR"],
        "rw": sum(A.values()),
        # read/write/create files by exact name, but no directory listing and no execution
        "files_rw_nolist": A["WRITE_FILE"] | A["READ_FILE"] | A["MAKE_REG"] | A["MAKE_DIR"] | A["REMOVE_FILE"] |
                           A["TRUNCATE"] | A["REFER"]}


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]


NET_CONNECT_TCP = 1 << 1


class PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def abi_version(libc):
    return libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(1))


def handled_bits(abi):
    bits = 13
    bits += abi >= 2
    bits += abi >= 3
    bits += abi >= 5
    return (1 << bits) - 1


def restrict(rules, deny_net=False):
    """deny_net=True (ABI >= 4) additionally blocks every outgoing TCP connection: used for offline start-up dry runs."""
    libc = ctypes.CDLL(None, use_errno=True)
    abi = abi_version(libc)
    if abi < 3 or (deny_net and abi < 4):
        raise RuntimeError(f"Landlock unavailable or too old (ABI {abi})")
    handled = handled_bits(abi)
    attr = RulesetAttr(handled, NET_CONNECT_TCP if deny_net else 0)
    fd = libc.syscall(SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    applied = []
    for r in rules:
        path, access = r["path"], SETS[r["access"]] & handled
        if not os.path.exists(path):
            continue
        if not stat.S_ISDIR(os.stat(path).st_mode):
            access &= FILE_OK
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        pb = PathBeneath(access, pfd)
        if libc.syscall(SYS_ADD_RULE, fd, ctypes.c_uint32(RULE_PATH_BENEATH), ctypes.byref(pb), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
        os.close(pfd)
        applied.append((path, r["access"]))
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
        raise OSError(ctypes.get_errno(), "prctl")
    if libc.syscall(SYS_RESTRICT, fd, ctypes.c_uint32(0)) < 0:
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(fd)
    return abi, applied


def main():
    spec_path, sep, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
    assert sep == "--" and cmd, __doc__
    spec = json.load(open(spec_path))
    restrict(spec["rules"], deny_net=bool(spec.get("deny_net", False)))
    env = dict(spec.get("env", {}))
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
