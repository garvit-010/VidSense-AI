from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document


class YouTubeTranscriptLoader:
    def __init__(self, video_id: str, language: str = "en"):
        self.video_id = video_id
        self.language = language
        self.api = YouTubeTranscriptApi()

    def load(self) -> list[Document]:
        """
        Fetch YouTube transcript and return LangChain Documents
        """
        try:
            transcript = self.api.fetch(
                self.video_id,
                languages=[self.language]
            )
        except TranscriptsDisabled:
            raise ValueError("Transcript is disabled for this video")
        except Exception as e:
            raise ValueError(f"Failed to fetch transcript: {str(e)}")

        # transcript snippets are objects, not dicts
        full_text = " ".join(chunk.text for chunk in transcript)

        document = Document(
            page_content=full_text,
            metadata={
                "video_id": self.video_id,
                "source": "youtube"
            }
        )

        return [document]
