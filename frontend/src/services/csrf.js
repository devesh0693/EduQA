/**
 * Utility functions for handling CSRF tokens
 */

/**
 * Get CSRF token from cookies
 * @returns {string|null} CSRF token or null if not found
 */
export const getCSRFToken = () => {
  // Get all cookies
  const cookies = document.cookie.split(';');
  
  // Find the CSRF token cookie
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === 'csrftoken') {
      return decodeURIComponent(value);
    }
  }
  
  return null;
};

/**
 * Ensure we have a valid CSRF token by making a GET request if needed
 * @returns {Promise<string|null>} CSRF token or null if not available
 */
export const ensureCSRFToken = async () => {
  // First check if we already have a token
  let token = getCSRFToken();
  
  // If no token, try to get one by making a GET request
  if (!token) {
    try {
      await fetch('/api/csrf/', {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      // Try to get the token again after the request
      token = getCSRFToken();
    } catch (error) {
      console.error('Failed to get CSRF token:', error);
    }
  }
  
  return token;
};

/**
 * Get CSRF token, ensuring we have one first
 * @returns {Promise<string|null>} CSRF token or null if not available
 */
export const getCSRFTokenAsync = async () => {
  return await ensureCSRFToken();
};
