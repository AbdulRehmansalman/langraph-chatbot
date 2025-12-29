import React from 'react';
import type { ChatResponse } from '../../types';

interface MessageBubbleProps {
  message: ChatResponse;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="space-y-4">
      {/* User Message */}
      <div className="flex justify-end">
        <div className="max-w-[85%] md:max-w-3xl">
          <div className="bg-primary-500 text-white rounded-2xl rounded-tr-md px-3 py-2 md:px-4 md:py-3">
            <p className="whitespace-pre-wrap break-words text-sm md:text-base">
              {message.user_message}
            </p>
          </div>
          <div className="text-xs text-gray-500 text-right mt-1">
            {formatTimestamp(message.created_at)}
          </div>
        </div>
      </div>

      {/* Bot Response */}
      <div className="flex justify-start">
        <div className="max-w-[90%] md:max-w-3xl">
          <div className="flex items-start space-x-2 md:space-x-3">
            <div className="flex-shrink-0 w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-sm md:text-base">
              🤖
            </div>
            <div className="flex-1 min-w-0">
              <div className="bg-gray-100 rounded-2xl rounded-tl-md px-3 py-2 md:px-4 md:py-3">
                <p className="whitespace-pre-wrap break-words text-gray-800 text-sm md:text-base">
                  {message.bot_response}
                </p>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                AI Assistant • {formatTimestamp(message.created_at)}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
