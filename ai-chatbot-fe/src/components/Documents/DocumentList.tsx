import React from 'react';
import { useDocumentStore } from '../../stores/documentStore';
import { useDocuments, useDeleteDocument } from '../../hooks/queries';

const DocumentList: React.FC = () => {
  const { selectedDocuments, toggleDocumentSelection } = useDocumentStore();
  const { data: documents = [], isLoading: loading, error } = useDocuments();
  const deleteDocumentMutation = useDeleteDocument();

  const handleDelete = (documentId: string, filename: string) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    deleteDocumentMutation.mutate(documentId);
  };

  const getFileIcon = (contentType: string) => {
    if (contentType.includes('pdf')) return '📄';
    if (contentType.includes('text')) return '📝';
    if (contentType.includes('word')) return '📋';
    return '📄';
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
        <span className="ml-3 text-gray-600">Loading documents...</span>
      </div>
    );
  }

  if (error) {
    return <div className="error-message">{error.message}</div>;
  }

  if (documents.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-xl">
        <div className="text-6xl mb-4">📚</div>
        <h3 className="text-lg font-medium text-gray-700 mb-2">No documents uploaded yet</h3>
        <p className="text-gray-500">Upload your first document to start chatting!</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-800">Your Documents ({documents.length})</h3>
        {selectedDocuments.length > 0 && (
          <span className="text-sm text-primary-600 bg-primary-50 px-3 py-1 rounded-full">
            {selectedDocuments.length} selected for chat
          </span>
        )}
      </div>

      <div className="grid gap-4">
        {documents.map(doc => (
          <div
            key={doc.id}
            className={`
              card transition-all duration-200 hover:shadow-lg cursor-pointer
              ${
                selectedDocuments.includes(doc.id)
                  ? 'ring-2 ring-primary-500 bg-primary-50'
                  : 'hover:shadow-md'
              }
            `}
            onClick={() => toggleDocumentSelection(doc.id)}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1 min-w-0">
                <span className="text-2xl flex-shrink-0">{getFileIcon(doc.content_type)}</span>
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium text-gray-800 truncate" title={doc.filename}>
                    {doc.filename}
                  </h4>
                  <p className="text-sm text-gray-500">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </p>
                  <div className="flex items-center mt-1">
                    <span
                      className={`
                        text-xs px-2 py-1 rounded-full font-medium
                        ${
                          doc.processed
                            ? 'bg-green-100 text-green-700'
                            : 'bg-yellow-100 text-yellow-700'
                        }
                      `}
                    >
                      {doc.processed ? '✓ Processed' : '⏳ Processing...'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-2 flex-shrink-0">
                <button
                  onClick={e => {
                    e.stopPropagation();
                    toggleDocumentSelection(doc.id);
                  }}
                  className={`
                    w-8 h-8 rounded-lg flex items-center justify-center transition-colors
                    ${
                      selectedDocuments.includes(doc.id)
                        ? 'bg-primary-500 text-white'
                        : 'bg-gray-100 text-gray-600 hover:bg-primary-100 hover:text-primary-600'
                    }
                  `}
                  title={selectedDocuments.includes(doc.id) ? 'Deselect' : 'Select for chat'}
                >
                  {selectedDocuments.includes(doc.id) ? '✓' : '+'}
                </button>
                <button
                  onClick={e => {
                    e.stopPropagation();
                    handleDelete(doc.id, doc.filename);
                  }}
                  className="w-8 h-8 rounded-lg flex items-center justify-center bg-gray-100 text-gray-600 hover:bg-red-100 hover:text-red-600 transition-colors"
                  title="Delete document"
                >
                  🗑️
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DocumentList;
