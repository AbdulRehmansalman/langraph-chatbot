import React, { Suspense, lazy, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLogout } from '../../hooks/queries';
import { useSEO } from '../../hooks/useSEO';
import { useAuthStore } from '../../stores/authStore';
import { useDocumentStore } from '../../stores/documentStore';
import { PageLoader } from '../LoadingSpinner';

// Lazy load heavy dashboard components
const DocumentUpload = lazy(() => import('../Documents/DocumentUpload'));
const DocumentList = lazy(() => import('../Documents/DocumentList'));
const ChatInterface = lazy(() => import('../Chat/ChatInterface'));
const GoogleCalendarStatus = lazy(() => import('../GoogleCalendar/GoogleCalendarStatus'));

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'chat' | 'documents'>('chat');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navigate = useNavigate();
  const { user } = useAuthStore();
  const { selectedDocuments, clearSelection } = useDocumentStore();
  const logoutMutation = useLogout();

  // Close sidebar on mobile by default
  React.useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarOpen(false);
      }
    };

    // Check on mount
    handleResize();

    // Optional: listen to resize events
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useSEO({
    title: 'Dashboard - AI Chatbot',
    description:
      'Access your AI-powered document chat dashboard. Upload documents, analyze content, and get intelligent insights through our advanced AI assistant.',
    keywords: 'dashboard, AI chatbot, document analysis, chat interface, AI assistant',
    ogTitle: 'Dashboard - AI Chatbot',
    ogDescription:
      'Access your AI-powered document chat dashboard. Upload documents, analyze content, and get intelligent insights through our advanced AI assistant.',
    canonical: 'https://ai-chatbot.example.com/dashboard',
  });

  const handleLogout = () => {
    logoutMutation.mutate();
  };

  const tabs = [
    { id: 'chat' as const, name: 'Chat', icon: '💬' },
    { id: 'documents' as const, name: 'Documents', icon: '📄' },
  ];

  return (
    <div className="h-[100dvh] flex bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <div
        className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 bg-white border-r border-gray-200 flex flex-col fixed left-0 top-0 h-[100dvh] z-40 overflow-hidden`}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-bold text-gray-800">AI Chatbot</h1>
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden p-1 text-gray-500 hover:text-gray-700"
            >
              ✕
            </button>
          </div>
          <p className="text-sm text-gray-600 mt-1">Welcome back, {user?.full_name}!</p>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-b-2 border-primary-500 text-primary-600 bg-primary-50'
                  : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-4">
          <Suspense fallback={<PageLoader text="Loading..." />}>
            {activeTab === 'documents' ? (
              <div className="space-y-6">
                <DocumentUpload />
                <DocumentList />
              </div>
            ) : (
              <div className="space-y-4">
                {selectedDocuments.length > 0 && (
                  <div className="bg-primary-50 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-medium text-primary-800">Documents Selected</h3>
                      <button
                        onClick={clearSelection}
                        className="text-primary-600 hover:text-primary-800 text-sm underline"
                      >
                        Clear all
                      </button>
                    </div>
                    <p className="text-sm text-primary-600 mt-1">
                      {selectedDocuments.length} document{selectedDocuments.length !== 1 ? 's' : ''}{' '}
                      selected for chat
                    </p>
                  </div>
                )}

                <div className="text-center py-8">
                  <div className="text-4xl mb-4">🤖</div>
                  <h3 className="font-medium text-gray-700 mb-2">Ready to Chat</h3>
                  <p className="text-gray-500 text-sm">
                    {selectedDocuments.length > 0
                      ? 'Ask questions about your selected documents'
                      : 'Select documents from the Documents tab or ask general questions'}
                  </p>
                </div>
              </div>
            )}
          </Suspense>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 space-y-2">
          <Suspense fallback={<div className="h-20 animate-pulse bg-gray-100 rounded-lg" />}>
            <GoogleCalendarStatus />
          </Suspense>
          <button
            onClick={() => navigate('/change-password')}
            className="w-full flex items-center justify-center px-4 py-2 text-sm text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <span className="mr-2">🔒</span>
            Change Password
          </button>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center px-4 py-2 text-sm text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <span className="mr-2">🚪</span>
            Logout
          </button>
        </div>
      </div>

      {/* Sidebar Toggle Button - Desktop */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className={`hidden lg:flex fixed top-1/2 transform -translate-y-1/2 bg-white border border-gray-200 rounded-full p-2 shadow-md hover:bg-gray-50 transition-all duration-300 z-50 ${
          sidebarOpen ? 'left-80 -ml-4' : 'left-0 ml-4'
        }`}
        aria-label={sidebarOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        {sidebarOpen ? (
          <svg
            className="w-4 h-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
        ) : (
          <svg
            className="w-4 h-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        )}
      </button>

      {/* Main Content */}
      <div
        className={`flex-1 flex flex-col h-[100dvh] overflow-hidden ${sidebarOpen ? 'ml-80' : 'ml-0'} transition-all duration-300`}
      >
        {/* Mobile Header */}
        <div className="lg:hidden bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label="Open menu"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>
            <h1 className="text-lg font-semibold text-gray-800">AI Chatbot</h1>
            <button
              onClick={handleLogout}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              aria-label="Logout"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="flex-1 bg-white overflow-hidden">
          <Suspense fallback={<PageLoader text="Loading chat..." />}>
            <ChatInterface />
          </Suspense>
        </div>
      </div>

      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

export default Dashboard;
