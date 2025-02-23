import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [conversation, setConversation] = useState<
    { type: "user" | "ai"; text: string; isLoading?: boolean }[]
  >([]);
  const [isLoading, setIsLoading] = useState(false);
  const conversationRef = useRef<HTMLDivElement>(null);

  const generateCoverLetter = async () => {
    if (!jobDescription.trim()) return;

    setConversation((prev) => [
      ...prev,
      { type: "user", text: jobDescription },
      { type: "ai", text: "", isLoading: true },
    ]);
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8080/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ job_description: jobDescription }),
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;

        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const word = line.replace("data: ", "").trim();
              if (word === "") {
                setConversation((prev) =>
                  prev.map((item, idx) =>
                    idx === prev.length - 1
                      ? { ...item, isLoading: false }
                      : item
                  )
                );
                setIsLoading(false);
              } else {
                setConversation((prev) =>
                  prev.map((item, idx) =>
                    idx === prev.length - 1
                      ? { ...item, text: item.text + word + " " }
                      : item
                  )
                );
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Error generating cover letter:", error);
      setConversation((prev) =>
        prev.map((item, idx) =>
          idx === prev.length - 1 ? { ...item, isLoading: false } : item
        )
      );
      setIsLoading(false);
    }

    setJobDescription("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      generateCoverLetter();
    }
  };

  useEffect(() => {
    if (conversationRef.current) {
      conversationRef.current.scrollTop = conversationRef.current.scrollHeight;
    }
  }, [conversation]);

  return (
    <div className="chat-wrapper">
      <div className="chat-container">
        <div className="chat-header">
          <h1>Cover Letter Generator</h1>
        </div>
        <div className="chat-messages" ref={conversationRef}>
          {conversation.length === 0 && (
            <div className="chat-empty">
              Start by pasting a job description below!
            </div>
          )}
          {conversation.map((msg, index) => (
            <div
              key={index}
              className={`chat-message ${msg.type} ${
                msg.isLoading ? "loading" : ""
              }`}
            >
              <div className="message-content">{msg.text}</div>
            </div>
          ))}
        </div>
        <div className="chat-input">
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Paste job description here..."
            rows={3}
            disabled={isLoading}
          />
          <button onClick={generateCoverLetter} disabled={isLoading}>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              width="24"
              height="24"
            >
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
