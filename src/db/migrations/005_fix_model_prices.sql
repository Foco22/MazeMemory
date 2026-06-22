-- Fix Haiku 4.5 pricing (was incorrectly set to Haiku 3.5 rates).
-- Add cache_hit pricing for Anthropic models.

UPDATE model_prices SET
  input_usd_1m     = 10.00,
  output_usd_1m    = 50.00,
  cache_hit_usd_1m = 1.00
WHERE model = 'claude-fable-5';

UPDATE model_prices SET
  cache_hit_usd_1m = 0.30
WHERE model = 'claude-sonnet-4-6';

UPDATE model_prices SET
  input_usd_1m     = 1.00,
  output_usd_1m    = 5.00,
  cache_hit_usd_1m = 0.10
WHERE model = 'claude-haiku-4-5-20251001';
