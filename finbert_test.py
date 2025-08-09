import sentiment_scorer

sentiment_scorer.finbert_init()
bad_results = sentiment_scorer.finbert_scorer("Growth, Crash, Failure, Profit, Loss, Decrease")
good_results = sentiment_scorer.finbert_scorer("Up, Rise, success, increase, boom, growth")
print(bad_results)
print(good_results)
