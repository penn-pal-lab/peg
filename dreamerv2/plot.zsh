#!/usr/bin/zsh
# Maze plots
# echo "Maze: All Goals"
# python3 common/plot.py --indir ~/page_logdir_old/expt_maze/ --outdir "~/plotdir/maze/goal_0" --xaxis step --yaxis "max_eval_metric_success_cell/goal_0" --bins 5e4 --methods   ^page$   ^mega$   ^skewfit$ ^lexa$ ^p2e$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 600000 --labels "page" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"

 # Walker plots
# echo "Walker: All Goals"
# python3 common/plot.py --indir ~/page_logdir_old/expt_walker/ --outdir "~/plotdir/walkerhard/goal_all" --xaxis step --yaxis "max_eval_metric_success/goal_all" --bins 1e5 --methods   ^page$   ^mega$  ^skewfit$ ^lexa$ ^p2e$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 100000 750000 --labels "page" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"



#### Ant plots
# echo "Ant: All Goals"
# python3 common/plot.py --indir ~/page_logdir_walker_ant/expt_anthard/ --outdir "~/plotdir/anthard/goal_all" --xaxis step --yaxis "max_eval_metric_success/goal_all" --bins 5e4 --methods   ^page$   ^mega$ ^mega_statespace$ ^skewfit$ ^lexa$  ^p2e$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 950000 --ylim 0 0.9 --labels "page" "PEG" "mega" "MEGA" "mega_statespace" "MEGA-Unrestricted" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E" --palette contrast

# for i in {0..7}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py  --indir ~/page_logdir_walker_ant/expt_anthard/ --outdir "~/plotdir/anthard/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods   ^mppi2$   ^mega$ ^p2e$ ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000 --ylim 0 0.9
# done

#### Ant ablations
# echo "Ant Ablations: All Goals"
# python3 common/plot.py --indir ~/page_logdir_walker_ant/expt_anthard_ablations/ --outdir "~/plotdir/anthard_ablations/goal_all" --xaxis step --yaxis "max_eval_metric_success/goal_all" --bins 5e4 --methods   ^PEG$ ^sample_random_goal$ ^no_goal_optimization$ ^optimize_go_phase$ ^random_action_explore_phase$ ^no_explore_phase$   --add none --ylabel success --xlabel steps --size 8 5 --xlim 0 1000000  --ylim 0 0.9 --labels "sample_random_goal" "Rand-Goal PEG" "no_goal_optimization" "Seen-Goal PEG"  "optimize_go_phase" "Go-Value PEG" "random_action_explore_phase" "Rand-Explore PEG" "no_explore_phase" "No-Explore PEG"

# for i in {0..7}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py  --indir ~/page_logdir_walker_ant/expt_anthard/ --outdir "~/plotdir/anthard/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods   ^mppi2$   ^mega$ ^p2e$ ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000 --ylim 0 0.9
# done

## RND vs P2E ablation
# for i in {0..0}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/page_logdir_old/expt_ant/ --outdir "~/plotdir/ant_ablate/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^page-p2e$  ^page-rnd$  --add none --ylabel success --xlabel steps --size 4 4 --xlim 0 1000000  --smooth_factor 0.0
# done

# Stack plots
# echo "All Goals"
# python3 common/plot.py --indir ~/logdir/expt_stack/ --outdir "~/plotdir/stack2/goal_all" --xaxis step --yaxis "max_eval_metric_success/goal_all" --bins 2e4 --methods  ^page$   ^mega$  ^p2e$ ^example$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000

# for i in {0..7}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_stack/ --outdir "~/plotdir/stack2/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^page$   ^mega$  ^p2e$ ^example$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000
# done


# Unsup 3-stack plots
# echo "Unsup3stack All Goals"
# python3 common/plot.py --indir ~/logdir/expt_unsupstack2/ --outdir "~/plotdir/unsupstack2/goal_all" --xaxis step --yaxis "mean_eval_metric_success/goal_all" --bins 3e4 --methods ^page$   ^mega$  ^skewfit$ ^lexa$ ^p2e$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000

# for i in {2..2}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_unsupstack3/ --outdir "~/plotdir/unsupstack3/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^page$ ^mega$  ^skewfit$ ^lexa$ ^p2e$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000  --ylim 0 0.6 --smooth_factor 0.0 --labels "page" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"
# done

## CEM vs MPPI ablation
# for i in {2..2}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_unsupstack3/ --outdir "~/plotdir/unsupstack3/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^page-cem$  ^page-mppi$  --add none --ylabel success --xlabel steps --size 4 4 --xlim 0 1000000  --ylim 0 0.6 --smooth_factor 0.0
# done



# Kitchen plots
# echo "All Goals"
# python3 common/plot.py --indir ~/logdir/expt_kitchenhard/ --outdir "~/plotdir/kitchenhard/goal_all" --xaxis step --yaxis "max_eval_metric_success/goal_all" --bins 3e4 --methods ^page$   ^mega$  ^p2e$ ^skewfit$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000

# for i in {0..10}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_kitchenhard/ --outdir "~/plotdir/kitchenhard/goal_${GOAL}" --xaxis step --yaxis "max_eval_metric_success_task_relevant/goal_${GOAL}" --bins 3e4 --methods ^page$   ^mega$  ^p2e$ ^skewfit$  --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 1000000
# done

# New Unsup 3-stack plots
# echo "Unsup3stack All Goals"
# python3 common/plot.py --indir ~/logdir/expt_newstack3/ --outdir "~/plotdir/newstack3/goal_all" --xaxis step --yaxis "mean_eval_metric_success/goal_all" --bins 5e4 --methods ^mppi$ ^mega$  ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 900000  --ylim 0 1.0 --smooth_factor 0.0 --labels "mppi" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"

# for i in {9..14}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_newstack3/ --outdir "~/plotdir/newstack3/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^mppi$ ^mega$  ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 900000  --ylim 0 1.0 --smooth_factor 0.0 --labels "mppi" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"
# done

# Fixed Offline Eval Unsup 3-stack plots
echo "Unsup3stack All Goals"
python3 common/plot_3stack.py --indir ~/logdir/expt_newstack3/ --outdir "~/plotdir/newstack3/goal_all" --xaxis step --yaxis "mean_eval_metric_success/goal_all" --bins 5e4 --methods ^peg$ ^mega$  ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 900000  --ylim 0 100.0 --smooth_factor 0.0  --agg stderr1 --labels "peg" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"

# for i in {9..14}
# do
# echo "Goal: $i"
# GOAL=$i;
# python3 common/plot.py --indir ~/logdir/expt_newstack3/ --outdir "~/plotdir/newstack3/goal_${GOAL}" --xaxis step --yaxis "mean_eval_metric_success/goal_${GOAL}" --bins 5e4 --methods ^mppi$ ^mega$  ^skewfit$ --add none --ylabel success --xlabel steps --size 7 7 --xlim 0 900000  --ylim 0 1.0 --smooth_factor 0.0 --labels "mppi" "PEG" "mega" "MEGA" "skewfit" "Skewfit" "lexa" "LEXA" "p2e" "P2E"
# done