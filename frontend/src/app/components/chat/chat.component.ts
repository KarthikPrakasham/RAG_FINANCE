import { AfterViewChecked, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { ChatMessage, ChatRequest } from '../../models/chat-message.model';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent implements AfterViewChecked, OnInit {
  /** The conversation transcript rendered in the UI. */
  messages: ChatMessage[] = [
    {
      id: this.newId(),
      role: 'assistant',
      text: "Hi! I'm your Financial Regulations & Consumer Rights Assistant. Ask me about U.S. financial regulations, consumer rights, and fair lending.",
      timestamp: new Date(),
      status: 'done'
    }
  ];

  /** Two-way bound to the input box. */
  draft = '';

  /** True while we're waiting for an answer. */
  isThinking = false;

  @ViewChild('scrollAnchor') private scrollAnchor?: ElementRef<HTMLDivElement>;
  private shouldScroll = false;
  private activeSessionId: string | null = null;
  private readonly userId = 'frontend-user';

  /** Suggested prompts shown when the conversation is fresh. */
  readonly suggestions = [
    'Who is protected under ECOA?',
    'What is consumer credit?',
    'How do banks ensure fair lending compliance?',
    'What is fair lending?'
  ];

  constructor(private chat: ChatService) {}

  ngOnInit(): void {
    this.createSession();
  }

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollAnchor?.nativeElement.scrollIntoView({ behavior: 'smooth' });
      this.shouldScroll = false;
    }
  }

  /** Reset the conversation and create a fresh session on the backend. */
  newChat(): void {
    this.createSession(() => {
      this.messages = [
        {
          id: this.newId(),
          role: 'assistant',
          text: "Hi! I'm your Financial Regulations & Consumer Rights Assistant. Ask me about U.S. financial regulations, consumer rights, and fair lending.",
          timestamp: new Date(),
          status: 'done'
        }
      ];
      this.draft = '';
      this.isThinking = false;
    });
  }

  useSuggestion(text: string): void {
    this.draft = text;
    this.send();
  }

  /** Handle the send action from the input box. */
  send(): void {
    const query = this.draft.trim();
    if (!query || this.isThinking) {
      return;
    }

    this.pushMessage({
      id: this.newId(),
      role: 'user',
      text: query,
      timestamp: new Date(),
      status: 'done'
    });

    this.draft = '';
    this.isThinking = true;

    // Start the latency clock as close to the request as possible.
    const startedAt = performance.now();

    if (!this.activeSessionId) {
      this.createSession(() => this.submitQuery(query, startedAt));
      return;
    }

    this.submitQuery(query, startedAt);
  }

  private createSession(onReady?: () => void): void {
    this.chat.newSession().subscribe({
      next: (session) => {
        this.activeSessionId = session.session_id;
        onReady?.();
      },
      error: (err) => {
        console.error('Failed to create new session', err);
        this.pushMessage({
          id: this.newId(),
          role: 'assistant',
          text: 'Sorry — I could not create a chat session. Please make sure the backend is running, then try again.',
          timestamp: new Date(),
          status: 'error'
        });
      }
    });
  }

  private submitQuery(query: string, startedAt: number): void {
    const request: ChatRequest = {
      query,
      user_id: this.userId,
      session_id: this.activeSessionId ?? '',
      role: 'user'
    };

    // Call the RAG backend: the query is embedded and matched against the
    // ChromaDB vector store, and the grounded answer + citations come back.
    this.chat.ask(request).subscribe({
      next: (response) => {
        const responseTimeMs = Math.round(performance.now() - startedAt);

        // Prefer token counts reported by the backend; fall back to a
        // character-based estimate so the UI always has something to show.
        const backendIn = response.usage?.inputTokens;
        const backendOut = response.usage?.outputTokens;
        const estimated = backendIn == null || backendOut == null;

        this.pushMessage({
          id: this.newId(),
          role: 'assistant',
          text: response.answer,
          timestamp: new Date(),
          status: 'done',
          sources: response.sources,
          metrics: {
            inputTokens: backendIn ?? this.estimateTokens(query),
            outputTokens: backendOut ?? this.estimateTokens(response.answer),
            responseTimeMs,
            estimated
          }
        });
        this.isThinking = false;
      },
      error: (err) => {
        this.pushMessage({
          id: this.newId(),
          role: 'assistant',
          text:
            'Sorry — I could not reach the policy service. Please make sure ' +
            'the backend is running, then try again.',
          timestamp: new Date(),
          status: 'error'
        });
        this.isThinking = false;
        console.error('Chat request failed', err);
      }
    });
  }

  private pushMessage(message: ChatMessage): void {
    this.messages.push(message);
    this.shouldScroll = true;
  }

  private newId(): string {
    return `${this.messages?.length ?? 0}-${Math.floor(Math.random() * 1e9)}`;
  }

  /**
   * Rough token estimate used when the backend doesn't report usage.
   * Uses the common ~4-characters-per-token heuristic for English text.
   */
  private estimateTokens(text: string): number {
    if (!text) {
      return 0;
    }
    return Math.max(1, Math.ceil(text.trim().length / 4));
  }

  /** Human-friendly latency label, e.g. "820 ms" or "1.4 s". */
  formatDuration(ms: number): string {
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
  }
}
