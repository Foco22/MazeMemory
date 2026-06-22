-- Sync model_prices table with current prices.json.
-- Uses UPSERT so it works whether rows exist or not.

INSERT INTO model_prices (model, input_usd_1m, output_usd_1m, cache_hit_usd_1m) VALUES
  ('gpt-4o',                         2.50,   10.00,  1.25),
  ('gpt-4o-mini',                    0.15,    0.60,  0.075),
  ('gpt-5',                          1.25,   10.00,  0.125),
  ('gpt-5.4',                        2.50,   15.00,  0.25),
  ('gpt-5.5',                        5.00,   30.00,  0.50),
  ('gemini/gemini-3.1-pro-preview',  2.00,   12.00,  0.40),
  ('gemini/gemini-2.0-flash',        0.075,   0.30,  NULL),
  ('claude-opus-4-8',                5.00,   25.00,  0.50),
  ('claude-sonnet-4-6',              3.00,   15.00,  0.30),
  ('claude-haiku-4-5-20251001',      1.00,    5.00,  0.10),
  ('deepseek/deepseek-chat',         0.14,    0.28,  0.0028),
  ('deepseek/deepseek-v4-flash',     0.14,    0.28,  0.0028),
  ('deepseek/deepseek-v3',           0.14,    0.28,  0.0028),
  ('deepseek/deepseek-v3.2',         0.14,    0.28,  0.0028),
  ('deepseek/deepseek-v4-pro',       0.435,   0.87,  0.0043)
ON CONFLICT (model) DO UPDATE SET
  input_usd_1m     = EXCLUDED.input_usd_1m,
  output_usd_1m    = EXCLUDED.output_usd_1m,
  cache_hit_usd_1m = EXCLUDED.cache_hit_usd_1m;
