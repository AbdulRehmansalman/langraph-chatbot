import { create } from 'zustand';

interface DocumentState {
  selectedDocuments: string[];
  toggleDocumentSelection: (documentId: string) => void;
  clearSelection: () => void;
}

export const useDocumentStore = create<DocumentState>((set, get) => ({
  selectedDocuments: [],

  toggleDocumentSelection: (documentId: string) => {
    const { selectedDocuments } = get();
    const isSelected = selectedDocuments.includes(documentId);

    if (isSelected) {
      set({
        selectedDocuments: selectedDocuments.filter(id => id !== documentId),
      });
    } else {
      set({
        selectedDocuments: [...selectedDocuments, documentId],
      });
    }
  },

  clearSelection: () => {
    set({ selectedDocuments: [] });
  },
}));
