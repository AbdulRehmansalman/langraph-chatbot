import React from 'react';
import { Link } from 'react-router-dom';
import { useSEO } from '../hooks/useSEO';

const NotFound: React.FC = () => {
  useSEO({
    title: '404 - Page Not Found | AI Chatbot',
    description:
      'The page you are looking for does not exist. Return to AI Chatbot dashboard to continue using our intelligent document assistant.',
    keywords: '404, page not found, AI chatbot',
    ogTitle: '404 - Page Not Found | AI Chatbot',
    ogDescription:
      'The page you are looking for does not exist. Return to AI Chatbot dashboard to continue using our intelligent document assistant.',
    canonical: 'https://ai-chatbot.example.com/404',
  });

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="text-center">
        <div className="text-9xl font-bold text-gray-200 mb-4">404</div>
        <h1 className="text-3xl font-bold text-gray-800 mb-4">Page Not Found</h1>
        <p className="text-gray-600 mb-8 max-w-md">
          The page you are looking for doesn't exist or has been moved. Let's get you back to the
          right place.
        </p>
        <div className="space-y-4">
          <Link
            to="/dashboard"
            className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Go to Dashboard
          </Link>
          <div className="text-sm text-gray-500">
            or{' '}
            <Link to="/login" className="text-blue-600 hover:text-blue-700 underline">
              return to login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
