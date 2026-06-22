-- Model pricing table (USD per 1M tokens).
-- Mirrors prices.json so cost queries can be done entirely in SQL.

CREATE TABLE model_prices (
    model           TEXT        PRIMARY KEY,
    input_usd_1m    NUMERIC(10, 4) NOT NULL,
    output_usd_1m   NUMERIC(10, 4) NOT NULL,
    cache_hit_usd_1m NUMERIC(10, 4)
);

INSERT INTO model_prices (model, input_usd_1m, output_usd_1m, cache_hit_usd_1m) VALUES
  ('claude-fable-5',                10.00,  50.00,  1.00),
  ('claude-sonnet-4-6',              3.00,  15.00,  0.30),
  ('claude-haiku-4-5-20251001',      1.00,   5.00,  0.10),
  ('gpt-5',                          1.25,  10.00,  NULL),
  ('gpt-5.5',                        5.00,  30.00,  NULL),
  ('gpt-4o',                         2.50,  10.00,  NULL),
  ('gpt-4o-mini',                    0.15,   0.60,  NULL),
  ('gemini/gemini-3.1-pro-preview',  2.00,  12.00,  NULL),
  ('gemini/gemini-2.0-flash',        0.075,  0.30,  NULL),
  ('deepseek/deepseek-chat',         0.14,   0.28,  0.0028),
  ('deepseek/deepseek-v4-flash',     0.14,   0.28,  0.0028),
  ('deepseek/deepseek-v3',           0.14,   0.28,  0.0028),
  ('deepseek/deepseek-v3.2',         0.14,   0.28,  0.0028),
  ('deepseek/deepseek-v4-pro',       0.435,  0.87,  0.0043);