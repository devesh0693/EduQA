import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/qa';
const CORE_API_BASE_URL = process.env.REACT_APP_CORE_API_URL || 'http://localhost:8000/api';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
  timeout: 30000, // 30 second timeout
  withCredentials: true, // Required for cookies/sessions
  xsrfCookieName: 'csrftoken',
  xsrfHeaderName: 'X-CSRFToken',
});

// Set default credentials mode for all requests
api.defaults.withCredentials = true;

// Helper function to get CSRF token from cookies
function getCSRFToken() {
  // Try to get from cookie
  const cookieMatch = document.cookie.match(/csrftoken=([^;]+)/);
  if (cookieMatch && cookieMatch[1]) {
    return cookieMatch[1];
  }
  return null;
}

// Request interceptor for logging and CSRF token handling
api.interceptors.request.use(
  async (config) => {
    // Don't log sensitive data in production
    if (process.env.NODE_ENV !== 'production') {
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data || '');
    }
    
    // Add CSRF token for all non-GET/HEAD/OPTIONS requests
    const method = config.method?.toLowerCase();
    if (['post', 'put', 'patch', 'delete'].includes(method)) {
      // Get existing CSRF token
      let csrfToken = getCSRFToken();
      
      // If no token, try to get one
      if (!csrfToken) {
        try {
          await api.get('/qa/csrf-token/', {
            withCredentials: true,
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache',
            }
          });
          csrfToken = getCSRFToken();
        } catch (error) {
          console.warn('Failed to get CSRF token:', error);
        }
      }
      
      // Add CSRF token to headers if we have one
      if (csrfToken) {
        config.headers['X-CSRFToken'] = csrfToken;
      } else if (process.env.NODE_ENV !== 'production') {
        console.warn('No CSRF token available for request');
      }
    }
    
    // Ensure credentials are sent with every request
    config.withCredentials = true;
    
    return config;
  },
  (error) => {
    if (process.env.NODE_ENV !== 'production') {
      console.error('Request error:', error);
    }
    return Promise.reject(error);
  }
);

// Response interceptor for error handling and logging
api.interceptors.response.use(
  (response) => {
    // Log successful responses in development
    if (process.env.NODE_ENV !== 'production') {
      console.log(`[API] ${response.status} ${response.config.method?.toUpperCase()} ${response.config.url}`);
    }
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle 401 Unauthorized responses
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the session or get a new CSRF token
        await api.get('/qa/csrf-token/');
        return api(originalRequest);
      } catch (refreshError) {
        console.error('Failed to refresh session:', refreshError);
        // Redirect to login or handle session expiration
        window.location.href = '/login';
      }
    }
    
    // Log detailed error in development
    if (process.env.NODE_ENV !== 'production') {
      console.error('API Error:', {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
        message: error.message,
      });
    }
    
    // Handle specific error statuses
    if (error.response) {
      switch (error.response.status) {
        case 403:
          console.error('Forbidden: You do not have permission to perform this action');
          break;
        case 404:
          console.error('Resource not found');
          break;
        case 500:
          console.error('Server error occurred');
          break;
        default:
          console.error(`HTTP error: ${error.response.status}`);
      }
    } else if (error.request) {
      console.error('No response received from server. Please check your connection.');
    } else {
      console.error('Request setup error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

// Ask a question
export const askQuestion = async (question, sessionId = null) => {
  console.log('[API] Asking question:', question);
  
  try {
    // Ensure we have a valid question
    if (!question || typeof question !== 'string' || !question.trim()) {
      throw new Error('Question cannot be empty');
    }
    
    // Prepare the request payload
    const payload = { 
      question: question.trim(),
      include_sources: true,
      include_confidence: true,
      include_processing_time: true
    };
    
    // Add session ID if provided
    if (sessionId) {
      payload.session_id = sessionId;
    }
    
    console.log('[API] Sending request to /qa/ask/ with payload:', payload);
    
    // Make the API request
    const response = await api.post('/qa/ask/', payload);
    
    // Validate response
    if (!response || !response.data) {
      console.error('[API] Invalid or empty response from server');
      throw new Error('No valid response received from server');
    }
    
    const data = response.data;
    console.log('[API] Received response:', data);
    
    // Format the response to ensure consistent structure
    const formattedResponse = {
      answer: data.answer || 'I apologize, but I couldn\'t generate an answer for your question at this time.',
      confidence: typeof data.confidence === 'number' 
        ? Math.min(Math.max(0, data.confidence), 1) // Ensure confidence is between 0 and 1
        : 0.8, // Default confidence if not provided
      sources: Array.isArray(data.sources) 
        ? data.sources
            .filter(source => source && (source.content || source.title || source.document_id))
            .map((source, index) => ({
              id: source.id || `src-${Date.now()}-${index}`,
              title: source.title || 'Document',
              content: source.content || '',
              source: source.source || 'Document',
              score: typeof source.score === 'number' 
                ? Math.min(Math.max(0, source.score), 1) 
                : 0.8, // Default score if not provided
              document_id: source.document_id || `doc-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              relevance: typeof source.relevance === 'string' 
                ? source.relevance 
                : (typeof source.score === 'number' 
                    ? `${Math.round(source.score * 100)}%` 
                    : '80%')
            }))
        : [],
      isFallback: Boolean(data.is_fallback || data.isFallback || false),
      processingTime: typeof data.processing_time === 'number' 
        ? data.processing_time 
        : (typeof data.processingTime === 'number' ? data.processingTime : 0),
      contextUsed: data.context_used || data.contextUsed || 'General knowledge',
      answerId: data.answer_id || data.answerId || `ans-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: data.timestamp || new Date().toISOString(),
      sessionId: data.session_id || data.sessionId || sessionId || `sess-${Date.now()}`
    };
    
    console.log('[API] Formatted response:', formattedResponse);
    return formattedResponse;
  } catch (error) {
    console.error('[API] Error asking question:', error);
    
    // Format a user-friendly error message
    let errorMessage = 'An error occurred while processing your question.';
    let errorDetails = null;
    let statusCode = null;
    
    // Handle different types of errors
    if (error.response) {
      // The request was made and the server responded with an error status
      const { status, data, headers } = error.response;
      statusCode = status;
      
      console.error(`[API] Server responded with status ${status}:`, data);
      console.error('[API] Error response headers:', headers);
      
      // Map common HTTP status codes to user-friendly messages
      const statusMessages = {
        400: 'Your request was invalid. Please try rephrasing your question.',
        401: 'Authentication required. Please log in and try again.',
        403: 'Access denied. You do not have permission to perform this action.',
        404: 'The requested resource was not found.',
        422: 'Unable to process your request. The data sent was invalid.',
        429: 'Too many requests. Please wait a moment and try again.',
        500: 'A server error occurred. Our team has been notified.',
        502: 'Bad gateway. The server received an invalid response.',
        503: 'The service is currently unavailable. Please try again later.',
        504: 'Gateway timeout. The server took too long to respond.'
      };
      
      errorMessage = statusMessages[status] || `Server error (${status}). Please try again.`;
      errorDetails = data?.detail || data?.message || (typeof data === 'string' ? data : JSON.stringify(data));
      
    } else if (error.request) {
      // The request was made but no response was received
      console.error('[API] No response received:', error.request);
      errorMessage = 'Unable to connect to the server. Please check your internet connection and try again.';
      errorDetails = 'Network error or server is not responding';
      
    } else if (error.code === 'ECONNABORTED') {
      // Request timeout
      console.error('[API] Request timed out');
      errorMessage = 'The request timed out. The server is taking too long to respond.';
      errorDetails = 'Request timeout';
      
    } else if (error.code === 'ERR_NETWORK') {
      // Network error
      console.error('[API] Network error:', error.message);
      errorMessage = 'Network error. Please check your internet connection.';
      errorDetails = error.message;
      
    } else if (error.message) {
      // Other errors with a message
      console.error('[API] Error:', error.message);
      errorMessage = error.message;
    }
    
    // Create a formatted error object with consistent structure
    const formattedError = new Error(errorMessage);
    formattedError.name = 'QuestionError';
    formattedError.details = errorDetails;
    formattedError.statusCode = statusCode;
    formattedError.originalError = error;
    formattedError.isUserFacing = true;
    
    // Log the full error for debugging
    console.error('[API] Formatted error:', formattedError);
    
    // Return a fallback response instead of throwing
    const fallbackResponse = {
      answer: errorMessage,
      confidence: 0,
      sources: [],
      isFallback: true,
      processingTime: 0,
      contextUsed: errorDetails || 'Error',
      answerId: `error-${Date.now()}`,
      error: formattedError
    };
    
    console.log('[API] Returning fallback response:', fallbackResponse);
    return fallbackResponse;
  }
}

// Get document by ID
export const getDocument = async (documentId) => {
  try {
    const response = await api.get(`/documents/${documentId}/`);
    return response.data;
  } catch (error) {
    console.error('Error fetching document:', error);
    return null;
  }
};

// Get session history
export const getSessionHistory = async (sessionId) => {
  try {
    const response = await api.get(`/qa/sessions/${sessionId}/history/`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get session history');
  }
};

// Submit feedback
export const submitFeedback = async (answerId, feedbackType, feedbackText = '') => {
  try {
    const response = await api.post('/qa/feedback/', {
      answer_id: answerId,
      feedback_type: feedbackType,
      feedback_text: feedbackText
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to submit feedback');
  }
};

// Health check
export const healthCheck = async () => {
  try {
    const response = await api.get('/qa/health/');
    return response.data;
  } catch (error) {
    throw new Error('Backend service is unavailable');
  }
};

// Search documents
export const searchDocuments = async (query) => {
  try {
    // Use a separate axios instance for core API calls with different base URL
    const coreApi = axios.create({
      baseURL: CORE_API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Requested-With': 'XMLHttpRequest',
      },
      timeout: 30000,
      withCredentials: true
    });
    
    const response = await coreApi.post('/core/documents/search/', { query });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Search failed');
  }
};

// Get all questions (for demo purposes)
export const getQuestions = async () => {
  try {
    const response = await api.get('/qa/popular-questions/');
    return response.data;
  } catch (error) {
    throw new Error('Failed to fetch questions');
  }
};

export default api;