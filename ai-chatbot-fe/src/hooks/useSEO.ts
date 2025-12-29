import { useEffect } from 'react';

interface SEOData {
  title?: string;
  description?: string;
  keywords?: string;
  ogTitle?: string;
  ogDescription?: string;
  ogImage?: string;
  twitterTitle?: string;
  twitterDescription?: string;
  canonical?: string;
}

export const useSEO = (seoData: SEOData) => {
  useEffect(() => {
    if (seoData.title) {
      document.title = seoData.title;

      const titleMeta = document.querySelector('meta[name="title"]');
      if (titleMeta) {
        titleMeta.setAttribute('content', seoData.title);
      }
    }

    if (seoData.description) {
      const descMeta = document.querySelector('meta[name="description"]');
      if (descMeta) {
        descMeta.setAttribute('content', seoData.description);
      }
    }

    if (seoData.keywords) {
      const keywordsMeta = document.querySelector('meta[name="keywords"]');
      if (keywordsMeta) {
        keywordsMeta.setAttribute('content', seoData.keywords);
      }
    }

    if (seoData.ogTitle) {
      const ogTitleMeta = document.querySelector('meta[property="og:title"]');
      if (ogTitleMeta) {
        ogTitleMeta.setAttribute('content', seoData.ogTitle);
      }
    }

    if (seoData.ogDescription) {
      const ogDescMeta = document.querySelector('meta[property="og:description"]');
      if (ogDescMeta) {
        ogDescMeta.setAttribute('content', seoData.ogDescription);
      }
    }

    if (seoData.ogImage) {
      const ogImageMeta = document.querySelector('meta[property="og:image"]');
      if (ogImageMeta) {
        ogImageMeta.setAttribute('content', seoData.ogImage);
      }
    }

    if (seoData.twitterTitle) {
      const twitterTitleMeta = document.querySelector('meta[property="twitter:title"]');
      if (twitterTitleMeta) {
        twitterTitleMeta.setAttribute('content', seoData.twitterTitle);
      }
    }

    if (seoData.twitterDescription) {
      const twitterDescMeta = document.querySelector('meta[property="twitter:description"]');
      if (twitterDescMeta) {
        twitterDescMeta.setAttribute('content', seoData.twitterDescription);
      }
    }

    if (seoData.canonical) {
      let canonicalLink = document.querySelector('link[rel="canonical"]');
      if (!canonicalLink) {
        canonicalLink = document.createElement('link');
        canonicalLink.setAttribute('rel', 'canonical');
        document.head.appendChild(canonicalLink);
      }
      canonicalLink.setAttribute('href', seoData.canonical);
    }
  }, [seoData]);
};

export const SEO_DEFAULTS = {
  title: 'AI Chatbot - Intelligent Document Assistant',
  description:
    'Advanced AI-powered chatbot that helps you interact with your documents intelligently. Upload, chat, and get insights from your files with our cutting-edge AI technology.',
  keywords:
    'AI chatbot, document assistant, artificial intelligence, document analysis, chat with documents, AI assistant, machine learning',
  ogTitle: 'AI Chatbot - Intelligent Document Assistant',
  ogDescription:
    'Advanced AI-powered chatbot that helps you interact with your documents intelligently. Upload, chat, and get insights from your files with our cutting-edge AI technology.',
  ogImage: 'https://ai-chatbot.example.com/og-image.jpg',
  twitterTitle: 'AI Chatbot - Intelligent Document Assistant',
  twitterDescription:
    'Advanced AI-powered chatbot that helps you interact with your documents intelligently. Upload, chat, and get insights from your files with our cutting-edge AI technology.',
  canonical: 'https://ai-chatbot.example.com/',
};
