import sentiment_scorer

sentiment_scorer.finbert_init()
bad_results = sentiment_scorer.finbert_scorer("Down, Falling, failure, decrease, crash, drop")
good_results = sentiment_scorer.finbert_scorer("Up, Rise, success, increase, boom, growth")
print(bad_results)
print(good_results)
