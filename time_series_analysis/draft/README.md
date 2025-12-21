     from estatecnica.time_series_analysis.draft.transforms import to_series, daily_to_weekly_and_yearly
     ```
     or adjust `sys.path` in a notebook to include the `time_series_analysis` folder
     and import from `draft`.
3. Run small checks:
   - Confirm `to_series` accepts both `Series` and `DataFrame` inputs.
   - Validate `stationary_test_adf` returns a structured result and does not raise
     unexpectedly for reasonable input.
   - Plot using `plots.visual_inspection` to ensure it returns Axes and displays.

Suggested next steps (after you review)
- If the helpers look good:
  1. Create a non-draft package module (e.g. `time_series_analysis/stats.py`)
     and copy the validated code there.
  2. Replace the implementation of the corresponding methods in the original
     `time_series_data_nature_discovery.py` with small delegating wrappers that
     call the new helpers. Keep the wrapper names the same to minimize breakage.
  3. Update `time_series_forecast.py` imports to use the new helpers (or the
     slimmed `TimeSeriesAnalyzer` if you convert it).
  4. Add unit tests in `tests/` to lock expected behavior.
- Consider adding a lightweight `__init__.py` to expose a stable API used by
  notebooks, and update the notebooks to import from `estatecnica.time_series_analysis`.

Notes & considerations
- These drafts aim to be conservative and non-invasive. I did not modify the
  original files in place — only created drafts for review.
- I intentionally made the helpers raise informative exceptions for invalid input,
  favoring explicit errors over silent failures.
- If you prefer, I can proceed next to wire these drafts into a namespaced
  package (move to final modules, update imports in notebooks), but I will wait
  for your approval before touching the original implementation.

If you'd like, I can:
- Run the incremental migration: create production modules and add delegating
  wrappers in the original class (keeps public API).
- Add a small test script under `draft/tests` that demonstrates usage with your
  AAPL sample data.

Tell me which next step you prefer and I will prepare the changes in the `draft`
area for you to review.
