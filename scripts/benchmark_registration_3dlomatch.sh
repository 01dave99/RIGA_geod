for N_POINTS in 5000 2500 1000 500 250
do
python registration/evaluate_registration.py --source_path ./snapshot/tdmatch_pointnet_ppf_test/3DLoMatch --benchmark 3DLoMatch --n_points $N_POINTS
done
