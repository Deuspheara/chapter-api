use anyhow::{Context, Result};
use std::path::Path;
use walkdir::WalkDir;
use crate::models::Chapter;
use crate::redis_client::RedisClient;

pub struct ChapterLoader {
    redis_client: RedisClient,
    chapters_dir: String,
}

impl ChapterLoader {
    pub fn new(redis_client: RedisClient, chapters_dir: String) -> Self {
        Self {
            redis_client,
            chapters_dir,
        }
    }

    pub async fn load_all_chapters(&self) -> Result<usize> {
        let chapters_path = Path::new(&self.chapters_dir);
        if !chapters_path.exists() {
            return Err(anyhow::anyhow!("Chapters directory does not exist: {}", self.chapters_dir));
        }

        let mut loaded_count = 0;
        
        for entry in WalkDir::new(chapters_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        {
            match self.load_chapter_file(entry.path()).await {
                Ok(_) => {
                    loaded_count += 1;
                    if loaded_count % 100 == 0 {
                        tracing::info!("Loaded {} chapters so far...", loaded_count);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to load chapter from {}: {}", entry.path().display(), e);
                }
            }
        }

        tracing::info!("Successfully loaded {} chapters into Redis", loaded_count);
        Ok(loaded_count)
    }

    async fn load_chapter_file(&self, file_path: &Path) -> Result<()> {
        let content = tokio::fs::read_to_string(file_path).await
            .context("Failed to read chapter file")?;
        
        let chapter: Chapter = serde_json::from_str(&content)
            .context("Failed to parse chapter JSON")?;
        
        self.redis_client.store_chapter(&chapter)
            .context("Failed to store chapter in Redis")?;
        
        Ok(())
    }

    pub async fn reload_chapters(&self) -> Result<usize> {
        tracing::info!("Flushing existing chapters and reloading...");
        self.redis_client.flush_chapters()?;
        self.load_all_chapters().await
    }
}