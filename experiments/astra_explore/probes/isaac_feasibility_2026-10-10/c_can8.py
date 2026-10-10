res = {}
res["X1_d6.8"] = grasp_trial3("X1", [-0.068, -0.042, 0.06], side_shift=0.03)
res["X2_d7.5"] = grasp_trial3("X2", [-0.075, -0.042, 0.06], side_shift=0.03)
res["X3_d6.8_press"] = grasp_trial3("X3", [-0.068, -0.036, 0.06], side_shift=0.035)
res["X4_d6.0"] = grasp_trial3("X4", [-0.060, -0.042, 0.06], side_shift=0.03)
print("SUMMARY", res)
