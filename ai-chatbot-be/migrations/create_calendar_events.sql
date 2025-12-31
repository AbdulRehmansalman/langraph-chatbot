-- Calendar Events Table for Appointment Scheduling
-- Run this in your Supabase SQL Editor

CREATE TABLE IF NOT EXISTS calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    duration_minutes INTEGER DEFAULT 60,
    description TEXT,
    location TEXT,
    participants TEXT[] DEFAULT '{}',
    status TEXT DEFAULT 'confirmed' CHECK (status IN ('confirmed', 'cancelled', 'pending', 'completed')),
    cancellation_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for faster queries by user and date
CREATE INDEX IF NOT EXISTS idx_calendar_events_user_id ON calendar_events(user_id);
CREATE INDEX IF NOT EXISTS idx_calendar_events_start_time ON calendar_events(start_time);
CREATE INDEX IF NOT EXISTS idx_calendar_events_user_start ON calendar_events(user_id, start_time);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE calendar_events ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to see only their own events (adjust as needed)
CREATE POLICY "Users can view own events" ON calendar_events
    FOR SELECT USING (true);  -- Adjust based on your auth setup

CREATE POLICY "Users can insert own events" ON calendar_events
    FOR INSERT WITH CHECK (true);  -- Adjust based on your auth setup

CREATE POLICY "Users can update own events" ON calendar_events
    FOR UPDATE USING (true);  -- Adjust based on your auth setup

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_calendar_events_updated_at
    BEFORE UPDATE ON calendar_events
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
