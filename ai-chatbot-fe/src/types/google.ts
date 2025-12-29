export interface GoogleAuthStatus {
  connected: boolean;
  expires_at?: string;
  scopes?: string[];
  message?: string;
}

export interface GoogleAuthResponse {
  auth_url: string;
  message: string;
  redirect_uri: string;
}

export interface GoogleDisconnectResponse {
  message: string;
}

export interface Meeting {
  id: string;
  user_id: string;
  title: string;
  scheduled_time: string;
  duration_minutes: number;
  google_event_id?: string;
  google_meet_link?: string;
  calendar_link?: string;
  status: 'scheduled' | 'completed' | 'cancelled';
  raw_request?: string;
  created_at: string;
  updated_at: string;
}

export interface MeetingScheduleResponse {
  type: 'success' | 'auth_required' | 'error';
  message: string;
  meeting?: Meeting;
  auth_url?: string;
  google_meet_link?: string;
  calendar_link?: string;
  instructions?: string[];
}
