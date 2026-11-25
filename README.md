# How Many Small Models Does It Take to Predict a Big One? — code and data

Code, executed notebooks and result data for *How Many Small Models Does It Take to Predict a Big One? Pre-Registered Tests and Identifiability Limits for Architecture-Effect Extrapolation* (Findings of EACL 2027).

The notebooks are stored *with* their outputs, so the numbers and plots are readable without re-running anything. Re-running needs the tokenised corpus, which is not included: `notebooks/prep_bpe_wt103.py` rebuilds it into `dataset/`. Run the notebooks from inside `notebooks/`. Everything ran on one 8 GB GPU with PyTorch 2.6 (CUDA 12.4); see `requirements.txt`.

## `notebooks/`

`model.py` is the transformer, and `sweep_utils.py` the training/eval harness of the baseline sweep.

    02, 03     baseline-recipe sweep on TinyShakespeare (Appendix B)
    22, 23     29.4M runs, seeds 0 and 1
    25, 26     step count and token budget (Section 4)
    27, 28     flat control: zero-initialised gated convolution (Section 4)
    29         retrospective test of two-rung screening (Section 4)
    30         matched-budget ladder (Table 1); commits the 91M forecasts
    31         Round 1 validation at 91.3M (Section 5.1)
    32         held-out intervention, multi-token prediction (Section 5.3)
    33         Round 2 validation at 162.9M (Section 5.4)
    34         2x token budget (Section 8)
    35         checks added after review (Appendix E)

TinyShakespeare is `data/tinyshakespeare/input.txt` from <https://github.com/karpathy/char-rnn>, saved as `dataset/tinyshakespeare/input.txt`.

## `results/`

Result data, one CSV row per trained model or per fitted forecast.

    preregistered_90M.csv        forecasts committed before any 91.3M model existed
    preregistered_160M.csv       forecasts committed before any 162.9M model existed
    preregistered_mtp_91M.csv    forecasts for the held-out intervention, committed
                                 before it was trained
    validation90_graded.csv      the same forecasts scored against what was measured
    validation160_graded.csv
    validation_mtp_graded.csv
    camready_runs.csv            official-layout Differential Attention and the
                                 learning-rate check (Appendix E)
    *_runs.csv                   per-experiment records, named for the notebook that
                                 produced them

Perplexities are validation perplexities read at the final step. A reported effect Δ is the mean of the two per-seed paired differences against the same-seed baseline, in percent, negative meaning the intervention helps.

The scripts compute the theory results (`theorem_threshold.py`, `theorem_unknown_alpha.py`, `propositions.py`, `theory_refit.py`, `camready_theory.py`), the Monte-Carlo verification (`mc_verify.py`), and Figures 1, 3 and 4 (`fig_naive_fail.py`, `fig_theory_refit.py`, `fig_disguise.py`).

## License

Code: MIT (`LICENSE`). WikiText-103 (CC BY-SA 3.0 / GFDL) is not redistributed here.
