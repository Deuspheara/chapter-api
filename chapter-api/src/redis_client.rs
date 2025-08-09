use anyhow::{Context, Result};
use redis::{Client, Connection, Commands};
use crate::models::Chapter;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct RedisClient {
    client: Client,
}

impl RedisClient {
    pub fn new(redis_url: &str) -> Result<Self> {
        let client = Client::open(redis_url)
            .context("Failed to create Redis client")?;
        Ok(Self { client })
    }

    pub fn get_connection(&self) -> Result<Connection> {
        self.client.get_connection()
            .context("Failed to get Redis connection")
    }

    pub fn store_chapter(&self, chapter: &Chapter) -> Result<()> {
        let mut conn = self.get_connection()?;
        let key = format!("chapter:{:05}", chapter.number);
        let json_data = serde_json::to_string(chapter)
            .context("Failed to serialize chapter")?;
        
        conn.set::<_, _, ()>(&key, json_data)
            .context("Failed to store chapter in Redis")?;
        
        tracing::info!("Stored chapter {} in Redis", chapter.number);
        Ok(())
    }

    pub fn get_chapter(&self, chapter_number: u32) -> Result<Option<Chapter>> {
        let mut conn = self.get_connection()?;
        let key = format!("chapter:{:05}", chapter_number);
        
        let json_data: Option<String> = conn.get(&key)
            .context("Failed to get chapter from Redis")?;
        
        match json_data {
            Some(data) => {
                let chapter = serde_json::from_str(&data)
                    .context("Failed to deserialize chapter")?;
                Ok(Some(chapter))
            }
            None => Ok(None),
        }
    }

    pub fn get_chapters_batch(&self, start: u32, end: u32) -> Result<Vec<Chapter>> {
        let mut conn = self.get_connection()?;
        let mut chapters = Vec::new();
        
        for chapter_num in start..=end {
            let key = format!("chapter:{:05}", chapter_num);
            if let Ok(Some(json_data)) = conn.get::<_, Option<String>>(&key) {
                if let Ok(chapter) = serde_json::from_str::<Chapter>(&json_data) {
                    chapters.push(chapter);
                }
            }
        }
        
        Ok(chapters)
    }

    pub fn get_total_chapters(&self) -> Result<u32> {
        let mut conn = self.get_connection()?;
        let keys: Vec<String> = conn.keys("chapter:*")
            .context("Failed to get chapter keys")?;
        Ok(keys.len() as u32)
    }

    pub fn flush_chapters(&self) -> Result<()> {
        let mut conn = self.get_connection()?;
        let keys: Vec<String> = conn.keys("chapter:*")
            .context("Failed to get chapter keys")?;
        
        if !keys.is_empty() {
            conn.del::<_, ()>(&keys)
                .context("Failed to delete chapter keys")?;
            tracing::info!("Flushed {} chapters from Redis", keys.len());
        }
        Ok(())
    }

    // RAG Response Caching
    pub fn cache_rag_response(&self, query_hash: &str, response: &str, ttl_seconds: u64) -> Result<()> {
        let mut conn = self.get_connection()?;
        let key = format!("rag_cache:{}", query_hash);
        conn.set_ex(&key, response, ttl_seconds)
            .context("Failed to cache RAG response")?;
        Ok(())
    }

    pub fn get_cached_rag_response(&self, query_hash: &str) -> Result<Option<String>> {
        let mut conn = self.get_connection()?;
        let key = format!("rag_cache:{}", query_hash);
        conn.get(&key)
            .context("Failed to get cached RAG response")
    }

    // Chapter Recap Caching
    pub fn cache_chapter_recap(&self, chapter_number: u32, recap: &str) -> Result<()> {
        let mut conn = self.get_connection()?;
        let key = format!("recap:{:05}", chapter_number);
        conn.set(&key, recap)
            .context("Failed to cache chapter recap")?;
        tracing::info!("Cached recap for chapter {}", chapter_number);
        Ok(())
    }

    pub fn get_cached_recap(&self, chapter_number: u32) -> Result<Option<String>> {
        let mut conn = self.get_connection()?;
        let key = format!("recap:{:05}", chapter_number);
        conn.get(&key)
            .context("Failed to get cached recap")
    }

    // Immersive Analysis Caching
    pub fn cache_mood_analysis(&self, chapter_number: u32, analysis: &str) -> Result<()> {
        let mut conn = self.get_connection()?;
        let key = format!("mood:{:05}", chapter_number);
        conn.set_ex(&key, analysis, 3600) // 1 hour TTL
            .context("Failed to cache mood analysis")?;
        Ok(())
    }

    pub fn get_cached_mood_analysis(&self, chapter_number: u32) -> Result<Option<String>> {
        let mut conn = self.get_connection()?;
        let key = format!("mood:{:05}", chapter_number);
        conn.get(&key)
            .context("Failed to get cached mood analysis")
    }

    pub fn cache_mystery_elements(&self, chapter_number: u32, elements: &str) -> Result<()> {
        let mut conn = self.get_connection()?;
        let key = format!("mystery:{:05}", chapter_number);
        conn.set_ex(&key, elements, 3600) // 1 hour TTL
            .context("Failed to cache mystery elements")?;
        Ok(())
    }

    pub fn get_cached_mystery_elements(&self, chapter_number: u32) -> Result<Option<String>> {
        let mut conn = self.get_connection()?;
        let key = format!("mystery:{:05}", chapter_number);
        conn.get(&key)
            .context("Failed to get cached mystery elements")
    }

    // Hash generation for cache keys
    pub fn generate_query_hash(&self, query: &str) -> String {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}