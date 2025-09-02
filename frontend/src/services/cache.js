// Cache service for storing search results and answers
class CacheService {
  constructor() {
    this.searchCacheKey = 'eduqa_search_cache';
    this.qaCacheKey = 'eduqa_qa_cache';
    this.maxCacheSize = 100; // Maximum number of cached items
    this.cacheExpiry = 1000 * 60 * 30; // 30 minutes
  }

  // Generate cache key from query
  generateKey(query) {
    return query.toLowerCase().trim().replace(/\s+/g, '_');
  }

  // Get cached search results
  getCachedSearch(query) {
    try {
      const cache = JSON.parse(localStorage.getItem(this.searchCacheKey) || '{}');
      const key = this.generateKey(query);
      const cachedItem = cache[key];

      if (cachedItem && (Date.now() - cachedItem.timestamp) < this.cacheExpiry) {
        console.log(`[Cache] Hit for search: "${query}"`);
        return cachedItem.data;
      }

      if (cachedItem) {
        // Remove expired cache
        delete cache[key];
        localStorage.setItem(this.searchCacheKey, JSON.stringify(cache));
      }

      return null;
    } catch (error) {
      console.error('[Cache] Error getting cached search:', error);
      return null;
    }
  }

  // Cache search results
  setCachedSearch(query, results) {
    try {
      const cache = JSON.parse(localStorage.getItem(this.searchCacheKey) || '{}');
      const key = this.generateKey(query);

      // Add new cache entry
      cache[key] = {
        data: results,
        timestamp: Date.now()
      };

      // Manage cache size
      const cacheKeys = Object.keys(cache);
      if (cacheKeys.length > this.maxCacheSize) {
        // Remove oldest entries
        const sortedKeys = cacheKeys.sort((a, b) => cache[a].timestamp - cache[b].timestamp);
        const keysToRemove = sortedKeys.slice(0, cacheKeys.length - this.maxCacheSize);
        keysToRemove.forEach(keyToRemove => delete cache[keyToRemove]);
      }

      localStorage.setItem(this.searchCacheKey, JSON.stringify(cache));
      console.log(`[Cache] Stored search: "${query}"`);
    } catch (error) {
      console.error('[Cache] Error caching search:', error);
    }
  }

  // Get cached QA answer
  getCachedAnswer(question) {
    try {
      const cache = JSON.parse(localStorage.getItem(this.qaCacheKey) || '{}');
      const key = this.generateKey(question);
      const cachedItem = cache[key];

      if (cachedItem && (Date.now() - cachedItem.timestamp) < this.cacheExpiry) {
        console.log(`[Cache] Hit for question: "${question}"`);
        return cachedItem.data;
      }

      if (cachedItem) {
        // Remove expired cache
        delete cache[key];
        localStorage.setItem(this.qaCacheKey, JSON.stringify(cache));
      }

      return null;
    } catch (error) {
      console.error('[Cache] Error getting cached answer:', error);
      return null;
    }
  }

  // Cache QA answer
  setCachedAnswer(question, answer) {
    try {
      const cache = JSON.parse(localStorage.getItem(this.qaCacheKey) || '{}');
      const key = this.generateKey(question);

      // Add new cache entry
      cache[key] = {
        data: answer,
        timestamp: Date.now()
      };

      // Manage cache size
      const cacheKeys = Object.keys(cache);
      if (cacheKeys.length > this.maxCacheSize) {
        // Remove oldest entries
        const sortedKeys = cacheKeys.sort((a, b) => cache[a].timestamp - cache[b].timestamp);
        const keysToRemove = sortedKeys.slice(0, cacheKeys.length - this.maxCacheSize);
        keysToRemove.forEach(keyToRemove => delete cache[keyToRemove]);
      }

      localStorage.setItem(this.qaCacheKey, JSON.stringify(cache));
      console.log(`[Cache] Stored answer: "${question}"`);
    } catch (error) {
      console.error('[Cache] Error caching answer:', error);
    }
  }

  // Clear all cache
  clearCache() {
    try {
      localStorage.removeItem(this.searchCacheKey);
      localStorage.removeItem(this.qaCacheKey);
      console.log('[Cache] All cache cleared');
    } catch (error) {
      console.error('[Cache] Error clearing cache:', error);
    }
  }

  // Clear expired cache entries
  cleanExpiredCache() {
    try {
      const now = Date.now();

      // Clean search cache
      const searchCache = JSON.parse(localStorage.getItem(this.searchCacheKey) || '{}');
      const validSearchEntries = {};
      Object.keys(searchCache).forEach(key => {
        if ((now - searchCache[key].timestamp) < this.cacheExpiry) {
          validSearchEntries[key] = searchCache[key];
        }
      });
      localStorage.setItem(this.searchCacheKey, JSON.stringify(validSearchEntries));

      // Clean QA cache
      const qaCache = JSON.parse(localStorage.getItem(this.qaCacheKey) || '{}');
      const validQaEntries = {};
      Object.keys(qaCache).forEach(key => {
        if ((now - qaCache[key].timestamp) < this.cacheExpiry) {
          validQaEntries[key] = qaCache[key];
        }
      });
      localStorage.setItem(this.qaCacheKey, JSON.stringify(validQaEntries));

      console.log('[Cache] Expired entries cleaned');
    } catch (error) {
      console.error('[Cache] Error cleaning expired cache:', error);
    }
  }

  // Get cache statistics
  getCacheStats() {
    try {
      const searchCache = JSON.parse(localStorage.getItem(this.searchCacheKey) || '{}');
      const qaCache = JSON.parse(localStorage.getItem(this.qaCacheKey) || '{}');

      return {
        searchEntries: Object.keys(searchCache).length,
        qaEntries: Object.keys(qaCache).length,
        totalEntries: Object.keys(searchCache).length + Object.keys(qaCache).length,
        maxSize: this.maxCacheSize,
        expiryMinutes: this.cacheExpiry / (1000 * 60)
      };
    } catch (error) {
      console.error('[Cache] Error getting cache stats:', error);
      return { searchEntries: 0, qaEntries: 0, totalEntries: 0 };
    }
  }
}

// Create and export singleton instance
const cacheService = new CacheService();

// Clean expired cache entries on initialization
cacheService.cleanExpiredCache();

export default cacheService;
