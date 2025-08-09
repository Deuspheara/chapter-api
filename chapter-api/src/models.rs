use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub index: u32,
    pub number: u32,
    pub title: String,
    pub url: String,
    pub word_count: u32,
    pub character_count: u32,
    pub paragraph_count: u32,
    pub reading_time_minutes: f64,
    pub language: String,
    pub content_hash: String,
    pub prev_chapter: Option<String>,
    pub next_chapter: Option<String>,
    pub scraped_at: String,
    // Removed content field - it's redundant with paragraphs
    pub paragraphs: Vec<String>,
    pub extraction_warnings: Option<serde_json::Value>,
}

// Lightweight version for batch operations
#[derive(Debug, Clone, Serialize)]
pub struct ChapterSummary {
    pub index: u32,
    pub number: u32,
    pub title: String,
    pub url: String,
    pub word_count: u32,
    pub character_count: u32,
    pub paragraph_count: u32,
    pub reading_time_minutes: f64,
    pub language: String,
    pub content_hash: String,
    pub prev_chapter: Option<String>,
    pub next_chapter: Option<String>,
    pub scraped_at: String,
    // No paragraphs for summary
}

#[derive(Debug, Serialize)]
pub struct ChapterBatch {
    pub chapters: Vec<Chapter>,
    pub total: usize,
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Serialize)]
pub struct ChapterBatchSummary {
    pub chapters: Vec<ChapterSummary>,
    pub total: usize,
    pub start: u32,
    pub end: u32,
}

impl From<Chapter> for ChapterSummary {
    fn from(chapter: Chapter) -> Self {
        Self {
            index: chapter.index,
            number: chapter.number,
            title: chapter.title,
            url: chapter.url,
            word_count: chapter.word_count,
            character_count: chapter.character_count,
            paragraph_count: chapter.paragraph_count,
            reading_time_minutes: chapter.reading_time_minutes,
            language: chapter.language,
            content_hash: chapter.content_hash,
            prev_chapter: chapter.prev_chapter,
            next_chapter: chapter.next_chapter,
            scraped_at: chapter.scraped_at,
        }
    }
}