# sweet.tv TV programme recommender (LLM4Rec final exam, AVITO group)

Aleksandr Selishchev. Kaggle competition: https://www.kaggle.com/competitions/sweettv-tv-program-recommender
(late submissions, metric MAP@5).

The task is to predict, for each of the 1,260 test users, five programmes (`tv_show_id`) they will watch in weeks 31-42.

## What is implemented

| Method | Where | Idea |
|---|---|---|
| Popularity baseline | `src/methods.py: popularity` | distinct users who watched the programme (>= 50% in one sitting) in the last 4 history weeks |
| Repeat count (reference) | `repeat_count` | how often the user watched the programme in the history window |
| Association rules | `association_rules` | item -> item rules over each user's set of watched programmes, confidence summed over the user's programmes; a -> a allowed |
| Personalised PageRank | `ppr` | random walk with restart on the programme co-view graph, restart = the user's own watch distribution |
| Implicit MF (ALS) | `als` | Hu-Koren-Volinsky ALS on the binary user x programme matrix |
| CBF, metadata | `cbf_metadata` | TF-IDF over category, genres, director, actors, decade; cosine to the user profile |
| CBF, schedule | `cbf_schedule` | programme = where and when it airs in the target window (channel x hour-of-week); user = viewing hours per channel x hour-of-week, unlabelled airtime included |
| **Final: hybrid** | `rrf` | weighted reciprocal-rank fusion of PPR and CBF-schedule, weight chosen on validation |

The rules below are the same for every method.

**Join and threshold.** A view is matched to every schedule row on the same channel (`vsetv_id` = `channel_id`) whose airing interval overlaps the view interval. The history counts a programme airing as watched if one single view covers at least 50% of it. Relevance in validation needs at least 80% in one single view.

**Unlabelled airtime.** Rows with `tv_show_id = 0` are not items. They still count as viewing context in CBF-schedule.

**Candidate pool and repeats.** The candidate pool is every labelled programme that airs in the target window, taken from the schedule only. Repeats of already-watched programmes are allowed.

**Short lists and cold start.** If a method returns fewer than 5 programmes, the list is filled from the popularity list. Users with no history in the window get the popularity list.

**Validation splits.** All methods use the last 10 training weeks as history.
- V0: history 11-20, validation 21-25.
- V1: history 16-25, validation 26-30. All hyperparameters are selected on V1.
- Final: history 21-30, test 31-42.

## Reproduce

1. Accept the competition rules on Kaggle and download the data.
2. Put the five CSV files into `data/`. That folder is not committed.
   ```
   data/dataset11-30.csv
   data/export_arh_11-20-final.csv
   data/export_arh_21-30-final.csv
   data/export_arh_31-42-final.csv
   data/submission.csv
   ```
3. Install the dependencies and run:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python src/run_all.py --data data --out .
   ```
   A full run takes about 8 minutes on 2 CPU cores, with a peak of about 4 GB RAM.

## Outputs

- `results/run_log.txt`: the full log, including every validation MAP@5 and runtime.
- `results/validation_map5.csv`: MAP@5 of every method and grid setting on V0 and V1.
- `results/summary.json`: data statistics, threshold table, selected settings, timings, environment and the two example users.
- `results/examples_two_users.csv`: Top-5 of every method for the two example users, with id, title and category. It covers the V1 validation window, compared with what the user watched, and the test window.
- `submissions/submission_<method>.csv`: one Kaggle file per method.
- `submissions/FINAL_submission.csv`: a byte-identical copy of the file behind the designated final Kaggle score.
