import React, { useState, useEffect, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { googleCalendarAPI } from '../../services/api';
import type { GoogleAuthStatus } from '../../types';

const GoogleCalendarStatus: React.FC = () => {
  const [isConnecting, setIsConnecting] = useState(false);
  const queryClient = useQueryClient();
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const {
    data: status,
    isLoading,
    error,
  } = useQuery<GoogleAuthStatus>({
    queryKey: ['google-calendar-status'],
    queryFn: googleCalendarAPI.getStatus,
    // Only refetch when window regains focus if not connecting
    refetchOnWindowFocus: !isConnecting,
    // Disable automatic refetching - we control it manually when connecting
    refetchInterval: false,
  });

  // Stop polling when connected
  useEffect(() => {
    if (status?.connected && isConnecting) {
      stopPolling();
      setIsConnecting(false);
    }
  }, [status?.connected, isConnecting]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, []);

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  };

  const startPolling = () => {
    // Clear any existing polling
    stopPolling();

    // Poll every 5 seconds (less aggressive)
    pollIntervalRef.current = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: ['google-calendar-status'] });
    }, 5000);

    // Stop polling after 2 minutes max
    pollTimeoutRef.current = setTimeout(() => {
      stopPolling();
      setIsConnecting(false);
    }, 120000);
  };

  const connectMutation = useMutation({
    mutationFn: googleCalendarAPI.startAuth,
    onSuccess: data => {
      window.open(data.auth_url, '_blank');
      setIsConnecting(true);
      startPolling();
    },
    onError: error => {
      console.error('Failed to start Google Calendar auth:', error);
      setIsConnecting(false);
      stopPolling();
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: googleCalendarAPI.disconnect,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['google-calendar-status'] });
    },
  });

  if (isLoading) {
    return (
      <div className="p-4 bg-white rounded-lg border">
        <div className="animate-pulse flex space-x-2 items-center">
          <div className="w-4 h-4 bg-gray-300 rounded-full"></div>
          <div className="h-4 bg-gray-300 rounded w-32"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-red-600 text-sm">Failed to load calendar status</p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-white rounded-lg border">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div
            className={`w-3 h-3 rounded-full ${status?.connected ? 'bg-green-500' : 'bg-gray-300'}`}
          />
          <div>
            <h3 className="font-medium text-gray-900">Google Calendar</h3>
            <p className="text-sm text-gray-500">
              {status?.connected
                ? 'Connected - You can schedule meetings'
                : 'Not connected - Connect to schedule meetings'}
            </p>
          </div>
        </div>

        {status?.connected ? (
          <button
            onClick={() => disconnectMutation.mutate()}
            disabled={disconnectMutation.isPending}
            className="px-3 py-1 text-sm text-red-600 border border-red-200 rounded hover:bg-red-50 disabled:opacity-50"
          >
            {disconnectMutation.isPending ? 'Disconnecting...' : 'Disconnect'}
          </button>
        ) : (
          <button
            onClick={() => connectMutation.mutate()}
            disabled={connectMutation.isPending || isConnecting}
            className="px-3 py-1 text-sm text-blue-600 border border-blue-200 rounded hover:bg-blue-50 disabled:opacity-50"
          >
            {connectMutation.isPending || isConnecting ? 'Connecting...' : 'Connect'}
          </button>
        )}
      </div>

      {isConnecting && (
        <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded">
          <p className="text-sm text-blue-700">
            Complete the authorization in the opened tab, then return here. This status will
            update automatically.
          </p>
        </div>
      )}

      {status?.connected && status.expires_at && (
        <div className="mt-2 text-xs text-gray-500">
          Connection expires: {new Date(status.expires_at).toLocaleDateString()}
        </div>
      )}
    </div>
  );
};

export default GoogleCalendarStatus;
