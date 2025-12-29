import React, { useState, useRef } from 'react';
import { useUploadDocument } from '../../hooks/queries';

const DocumentUpload: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [message, setMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadDocumentMutation = useUploadDocument();

  const handleFiles = (files: FileList) => {
    const file = files[0];
    if (!file) return;

    const allowedTypes = [
      'application/pdf',
      'text/plain',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    ];
    if (!allowedTypes.includes(file.type)) {
      setMessage('Please upload PDF, TXT, or DOCX files only.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      // 10MB limit
      setMessage('File size must be less than 10MB.');
      return;
    }

    setMessage('');

    uploadDocumentMutation.mutate(
      { file },
      {
        onSuccess: () => {
          setMessage(`File "${file.name}" uploaded successfully!`);
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        },
      }
    );
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="mb-8">
      <div
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300
          ${
            dragActive
              ? 'border-primary-500 bg-primary-50 scale-105'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }
          ${uploadDocumentMutation.isPending ? 'cursor-not-allowed opacity-70' : ''}
        `}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={!uploadDocumentMutation.isPending ? openFileDialog : undefined}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.docx"
          onChange={handleFileSelect}
          className="hidden"
        />

        {uploadDocumentMutation.isPending ? (
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
            <p className="text-gray-600">Uploading document...</p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="text-6xl">📄</div>
            <h3 className="text-xl font-semibold text-gray-700">Upload Document</h3>
            <p className="text-gray-500">Drag and drop your file here, or click to browse</p>
            <span className="text-sm text-gray-400">Supported: PDF, TXT, DOCX (max 10MB)</span>
          </div>
        )}
      </div>

      {(uploadDocumentMutation.error || message) && (
        <div
          className={`mt-4 ${uploadDocumentMutation.error ? 'error-message' : 'success-message'}`}
        >
          {uploadDocumentMutation.error?.message || message}
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
