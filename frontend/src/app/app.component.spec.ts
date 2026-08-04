/// <reference types="jasmine" />

import { TestBed } from '@angular/core/testing';
import { AppComponent } from './app.component';
import { ChatComponent } from './components/chat/chat.component';
import { of } from 'rxjs';

describe('AppComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [
        AppComponent
      ],
    }).compileComponents();
  });

  it('should create the app', () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app).toBeTruthy();
  });

  it(`should have as title 'hr-policy-chatbot'`, () => {
    const fixture = TestBed.createComponent(AppComponent);
    const app = fixture.componentInstance;
    expect(app.title).toEqual('hr-policy-chatbot');
  });

  it('should render title', () => {
    const fixture = TestBed.createComponent(AppComponent);
    fixture.detectChanges();
    const compiled = fixture.nativeElement as HTMLElement;
    expect(compiled.querySelector('.content span')?.textContent).toContain('hr-policy-chatbot app is running!');
  });
});

describe('ChatComponent', () => {
  it('creates a session when initialized', () => {
    const chatService = {
      newSession: jasmine.createSpy().and.returnValue(of({
        session_id: 'session-123',
        title: 'New Chat',
        created_at: '2026-08-04T00:00:00.000Z'
      }))
    } as any;

    const component = new ChatComponent(chatService);
    component.ngOnInit();

    expect(chatService.newSession).toHaveBeenCalled();
  });
});
