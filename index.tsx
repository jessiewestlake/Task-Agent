/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Chat, Type, GenerateContentConfig } from "@google/genai";
import { marked } from 'marked';
import DOMPurify from 'dompurify';

// --- TYPE DEFINITIONS ---
// Fix: Made uri and title optional to match the type from @google/genai library.
type GroundingChunkWeb = {
  uri?: string;
  title?: string;
}

// Fix: Made the `web` property optional to match the type from the `@google/genai` library, resolving a type error.
type GroundingChunk = {
  web?: GroundingChunkWeb;
};

type HistoryEntry = {
  role: 'user' | 'model';
  text: string;
  groundingChunks?: GroundingChunk[];
};

type Conversation = {
  id: string;
  title: string;
  history: HistoryEntry[];
  customInstructions: string;
  createdAt: number;
};

type TaskStatus = 'todo' | 'in-progress' | 'done';
type TaskPriority = 'low' | 'medium' | 'high';

type Task = {
  id: string;
  title: string;
  notes: string;
  status: TaskStatus;
  priority: TaskPriority;
  dueDate: string | null;
  createdAt: number;
  sourceConversationId: string | null;
  sourceMessageText: string | null;
};

type AppSettings = {
  nickname: string;
  customInstructions: string;
  agentName: string;
  agentAvatarUrl: string;
  searchArchived: boolean;
  isSidebarPinned: boolean;
};

type AppState = {
  conversations: Record<string, Conversation>;
  archivedConversations: Record<string, Conversation>;
  activeConversationId: string | null;
  tasks: Task[];
  settings: AppSettings;
};

type Command = {
  id: string;
  label: string;
  action: () => void;
  keywords?: string[];
};

type Tool = {
  id: 'web-search';
  label: string;
  icon: string;
};

type TaskSuggestion = {
    taskSuggested: boolean;
    taskTitle: string;
    taskNotes: string;
};

// --- CONSTANTS ---
const DEFAULT_AGENT_NAME = "Gemini";
const AGENT_AVATAR = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM7.07 18.28c.43-.9 3.05-1.78 4.93-1.78s4.5.88 4.93 1.78C15.57 19.36 13.86 20 12 20s-3.57-.64-4.93-1.72zm11.26-2.85c-1.29-1.24-4.07-2.13-6.33-2.13s-5.04.89-6.33 2.13C4.6 14.24 4 13.16 4 12c0-4.41 3.59-8 8-8s8 3.59 8 8c0 1.16-.6 2.24-1.67 3.43z"/><path d="M12 6c-1.94 0-3.5 1.56-3.5 3.5S10.06 13 12 13s3.5-1.56 3.5-3.5S13.94 6 12 6zm0 5c-.83 0-1.5-.67-1.5-1.5S11.17 8 12 8s1.5.67 1.5 1.5S12.83 11 12 11z"/></svg>`;
const USER_AVATAR = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>`;
const STATIC_SAMPLE_PROMPTS = [ "Explain quantum computing", "Tips for learning a new language", "Write a story about a robot who discovers music", "Framework vs. library" ];
const TYPING_INDICATOR_ID = 'typing-indicator-element';
const STORAGE_KEY = 'gemini-chat-app-state';
const availableTools: Tool[] = [
  {
    id: 'web-search',
    label: 'Search web',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>`
  }
];


// --- DOM ELEMENT REFERENCES ---
const appContainer = document.getElementById('app-container');
const conversationSidebar = document.getElementById('conversation-sidebar');
const mainContent = document.getElementById('main-content');
const sidebarToggleButton = document.getElementById('sidebar-toggle-button');
const pinSidebarButton = document.getElementById('pin-sidebar-button');
const newChatButton = document.getElementById('new-chat-button');
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const conversationList = document.getElementById('conversation-list');
const archivedSection = document.getElementById('archived-section');
const archivedList = document.getElementById('archived-list');
const settingsButton = document.getElementById('settings-button');
const themeSwitcher = document.getElementById('theme-switcher');

const chatView = document.getElementById('chat-view');
const chatTitle = document.getElementById('chat-title');
const welcomeScreen = document.getElementById('welcome-screen');
const samplePromptsContainer = document.getElementById('sample-prompts-container');
const chatHistoryContainer = document.getElementById('chat-history-container');
const chatHistoryElement = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form') as HTMLFormElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;
const sendButton = document.getElementById('send-button') as HTMLButtonElement;
const toolSelectorContainer = document.getElementById('tool-selector-container');
const toolChipsContainer = document.getElementById('tool-chips-container');


const commandPaletteOverlay = document.getElementById('command-palette-overlay');
const commandPaletteInput = document.getElementById('command-input') as HTMLInputElement;
const commandPaletteList = document.getElementById('command-list');

const settingsOverlay = document.getElementById('settings-overlay');
const settingsModal = document.getElementById('settings-modal');
const settingsForm = document.getElementById('settings-form') as HTMLFormElement;
const nicknameInput = document.getElementById('nickname-input') as HTMLInputElement;
const instructionsInput = document.getElementById('instructions-input') as HTMLTextAreaElement;
const agentNameInput = document.getElementById('agent-name-input') as HTMLInputElement;
const agentAvatarInput = document.getElementById('agent-avatar-input') as HTMLInputElement;
const searchArchivedCheckbox = document.getElementById('search-archived-checkbox') as HTMLInputElement;
const settingsSaveButton = document.getElementById('settings-save-button');
const settingsCloseButton = document.getElementById('settings-close-button');

const notificationContainer = document.getElementById('notification-container');

// Task Management DOM Elements
const tasksButton = document.getElementById('tasks-button');
const tasksOverlay = document.getElementById('tasks-overlay');
const tasksModal = document.getElementById('tasks-modal');
const closeTasksButton = document.getElementById('close-tasks-button');
const tasksList = document.getElementById('tasks-list');
const createNewTaskButton = document.getElementById('create-new-task-button');
const taskFormModal = document.getElementById('task-form-modal');
const taskForm = document.getElementById('task-form') as HTMLFormElement;
const taskFormTitle = document.getElementById('task-form-title');
const taskIdInput = document.getElementById('task-id-input') as HTMLInputElement;
const taskTitleInput = document.getElementById('task-title-input') as HTMLInputElement;
const taskNotesInput = document.getElementById('task-notes-input') as HTMLTextAreaElement;
const taskStatusSelect = document.getElementById('task-status-select') as HTMLSelectElement;
const taskPrioritySelect = document.getElementById('task-priority-select') as HTMLSelectElement;
const taskDueDateInput = document.getElementById('task-due-date-input') as HTMLInputElement;
const generateSubtasksButton = document.getElementById('generate-subtasks-button') as HTMLButtonElement;
const taskFormSourceLink = document.getElementById('task-form-source-link') as HTMLAnchorElement;
const taskDeleteButton = document.getElementById('task-delete-button') as HTMLButtonElement;
const taskCancelButton = document.getElementById('task-cancel-button');
const taskSaveButton = document.getElementById('task-save-button');


// --- STATE ---
let ai: GoogleGenAI;
let chat: Chat;
let state: AppState;
let activeCommandIndex = 0;
let activeTools = new Set<string>();
let editingTaskId: string | null = null;

// --- COMMANDS DEFINITION ---
const commands: Command[] = [
  { id: 'new-chat', label: 'New Chat', action: () => startNewChat(), keywords: ['clear', 'reset', 'start over'] },
  { id: 'toggle-theme', label: 'Toggle Light/Dark Mode', action: () => toggleTheme(), keywords: ['theme', 'dark', 'light', 'appearance'] },
  { id: 'open-settings', label: 'Open Settings', action: () => openSettings(), keywords: ['config', 'options', 'preferences'] },
  { id: 'open-tasks', label: 'Open Tasks', action: () => openTasksOverlay(), keywords: ['todo', 'list', 'tasks'] },
];

// --- STATE MANAGEMENT ---
function getInitialState(): AppState {
  return {
    conversations: {},
    archivedConversations: {},
    activeConversationId: null,
    tasks: [],
    settings: {
      nickname: 'You',
      customInstructions: '',
      agentName: DEFAULT_AGENT_NAME,
      agentAvatarUrl: '',
      searchArchived: false,
      isSidebarPinned: false,
    },
  };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function loadState() {
  const savedState = localStorage.getItem(STORAGE_KEY);
  if (savedState) {
    const parsedState = JSON.parse(savedState);
    state = { ...getInitialState(), ...parsedState };
    state.settings = { ...getInitialState().settings, ...parsedState.settings };
    state.tasks = parsedState.tasks || [];
  } else {
    state = getInitialState();
  }
}

// --- THEME ---
function applyTheme(theme: 'light' | 'dark') {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('gemini-theme', theme);
}

function toggleTheme() {
  const newTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  applyTheme(newTheme);
}

// --- INITIALIZATION ---
function initializeChat() {
  try {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  } catch (error) {
    console.error("Failed to initialize GoogleGenAI:", error);
    appendMessage('model', 'Error: Could not initialize AI. Please check your API key.');
  }
}

async function setupConversation() {
  const activeConvo = state.conversations[state.activeConversationId];
  if (!ai || !activeConvo) {
    chat = null;
    return;
  }
  
  const apiHistory = activeConvo.history.map(entry => ({
    role: entry.role,
    parts: [{ text: entry.text }],
  }));

  try {
    chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      history: apiHistory,
      config: activeConvo.customInstructions ? { systemInstruction: activeConvo.customInstructions } : {},
    });
  } catch (error) {
      console.error("Failed to create chat session:", error);
      appendMessage('model', 'Error: Could not start new chat session.');
  }
}

// --- UI RENDERING & MANIPULATION ---

function renderApp() {
  renderConversationList();
  if (state.activeConversationId && state.conversations[state.activeConversationId]) {
    renderActiveConversation();
  } else {
    showWelcomeScreen();
  }
  updateSidebarPinnedState(state.settings.isSidebarPinned);
}

function showWelcomeScreen() {
    chatHistoryElement.innerHTML = '';
    welcomeScreen.style.display = 'flex';
    chatHistoryElement.style.display = 'none';
    chatTitle.textContent = 'Gemini Chat';
    generateAndDisplaySamplePrompts();
}

function renderActiveConversation() {
    const convo = state.conversations[state.activeConversationId];
    if (!convo) {
        showWelcomeScreen();
        return;
    }
    welcomeScreen.style.display = 'none';
    chatHistoryElement.style.display = 'flex';
    chatTitle.textContent = convo.title;
    chatHistoryElement.innerHTML = '';
    convo.history.forEach(entry => appendMessage(entry.role, entry.text, entry.groundingChunks));
    setupConversation();
}

function createConversationListItem(convo: Conversation, isArchived: boolean): HTMLLIElement {
    const li = document.createElement('li');
    li.dataset.id = convo.id;
    li.tabIndex = 0;
    li.title = convo.title;
    
    const titleSpan = document.createElement('span');
    titleSpan.className = 'convo-title';
    titleSpan.textContent = convo.title;
    li.appendChild(titleSpan);

    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'convo-actions';

    const archiveButton = document.createElement('button');
    archiveButton.innerHTML = isArchived 
        ? `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M20.54 5.23a.9959.9959 0 0 0-1.41 0L12 12.59 4.87 5.46a.9959.9959 0 0 0-1.41 0L2.23 6.69c-.39.39-.39 1.02 0 1.41L10.59 16.5l-1.41 1.41a.9959.9959 0 0 0 0 1.41l1.18 1.18c.39.39 1.02.39 1.41 0L12 19.32l1.22 1.22c.39.39 1.02.39 1.41 0l1.18-1.18a.9959.9959 0 0 0 0-1.41L13.41 16.5l8.36-8.36c.39-.39.39-1.02 0-1.41l-1.23-1.49z"/></svg>`
        : `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M20.54 5.23a.9959.9959 0 0 0-1.41 0L12 12.59 4.87 5.46a.9959.9959 0 0 0-1.41 0L2.23 6.69c-.39.39-.39 1.02 0 1.41L10.59 16.5l-1.41 1.41a.9959.9959 0 0 0 0 1.41l1.18 1.18c.39.39 1.02.39 1.41 0L12 19.32l1.22 1.22c.39.39 1.02.39 1.41 0l1.18-1.18a.9959.9959 0 0 0 0-1.41L13.41 16.5l8.36-8.36c.39-.39.39-1.02 0-1.41l-1.23-1.49z"/></svg>`;
    archiveButton.ariaLabel = isArchived ? 'Unarchive' : 'Archive';
    archiveButton.onclick = (e) => { e.stopPropagation(); toggleArchiveConversation(convo.id); };
    
    const deleteButton = document.createElement('button');
    deleteButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/></svg>`;
    deleteButton.ariaLabel = 'Delete';
    deleteButton.onclick = (e) => { e.stopPropagation(); deleteConversation(convo.id, isArchived); };
    
    actionsDiv.appendChild(archiveButton);
    actionsDiv.appendChild(deleteButton);
    li.appendChild(actionsDiv);
    
    li.onclick = () => selectConversation(convo.id);
    li.onkeydown = (e) => { if(e.key === 'Enter' || e.key === ' ') selectConversation(convo.id); };

    return li;
}

function renderConversationList() {
    const filter = searchInput.value.toLowerCase();
    
    const renderList = (listEl: HTMLElement, convos: Record<string, Conversation>, isArchived: boolean) => {
        const sortedConvos = Object.values(convos).sort((a, b) => b.createdAt - a.createdAt);
        listEl.innerHTML = '';
        let hasVisibleItems = false;
        sortedConvos.forEach(convo => {
            const titleMatch = convo.title.toLowerCase().includes(filter);
            const contentMatch = state.settings.searchArchived || !isArchived 
                ? convo.history.some(h => h.text.toLowerCase().includes(filter))
                : false;
            
            if(titleMatch || contentMatch) {
                const li = createConversationListItem(convo, isArchived);
                if (convo.id === state.activeConversationId) {
                    li.classList.add('active');
                }
                listEl.appendChild(li);
                hasVisibleItems = true;
            }
        });
        return hasVisibleItems;
    };
    
    renderList(conversationList, state.conversations, false);
    const hasVisibleArchived = renderList(archivedList, state.archivedConversations, true);
    archivedSection.classList.toggle('hidden', !hasVisibleArchived);
}

function createSourcesElement(chunks: GroundingChunk[]): HTMLElement {
    const container = document.createElement('div');
    container.className = 'sources-container';

    const title = document.createElement('h3');
    title.className = 'sources-title';
    title.textContent = 'Sources';
    container.appendChild(title);

    const list = document.createElement('ol');
    list.className = 'sources-list';
    
    // Fix: Filter for unique sources by URI, ensuring chunks have a URI before processing.
    const uniqueSources = chunks.filter((chunk, index, self) =>
        chunk.web?.uri && index === self.findIndex((c) => (c.web?.uri === chunk.web.uri))
    );

    uniqueSources.forEach(chunk => {
        if (chunk.web && chunk.web.uri && chunk.web.title) {
            const item = document.createElement('li');
            item.className = 'source-item';
            
            const link = document.createElement('a');
            link.href = chunk.web.uri;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            
            const sourceTitle = document.createElement('span');
            sourceTitle.className = 'source-title';
            sourceTitle.textContent = chunk.web.title;
            
            const sourceUri = document.createElement('span');
            sourceUri.className = 'source-uri';
            sourceUri.textContent = chunk.web.uri;
            
            link.appendChild(sourceTitle);
            link.appendChild(sourceUri);
            item.appendChild(link);
            list.appendChild(item);
        }
    });
    container.appendChild(list);
    return container;
}


function appendMessage(role: 'user' | 'model', text: string, groundingChunks?: GroundingChunk[]): { messageGroup: HTMLElement; messageElement: HTMLElement } {
  const messageGroup = document.createElement('div');
  messageGroup.classList.add('message-group', role);

  const avatar = document.createElement('div');
  avatar.classList.add('avatar');
  if (role === 'user') {
    avatar.innerHTML = USER_AVATAR;
  } else {
    if (state.settings.agentAvatarUrl) {
        const img = document.createElement('img');
        img.src = state.settings.agentAvatarUrl;
        img.alt = 'Agent Avatar';
        img.onerror = () => { avatar.innerHTML = AGENT_AVATAR; }; // Fallback
        avatar.appendChild(img);
    } else {
        avatar.innerHTML = AGENT_AVATAR;
    }
  }

  const details = document.createElement('div');
  details.classList.add('details');

  const name = document.createElement('div');
  name.classList.add('name');
  name.textContent = role === 'user' ? state.settings.nickname : state.settings.agentName;

  const messageElement = document.createElement('div');
  messageElement.classList.add('message', `${role}-message`);
  
  const unsafeHtml = marked.parse(text) as string;
  messageElement.innerHTML = DOMPurify.sanitize(unsafeHtml);

  details.appendChild(name);
  details.appendChild(messageElement);
  
  if (role === 'model') {
      const actions = document.createElement('div');
      actions.className = 'message-actions';
      
      const addTaskButton = document.createElement('button');
      addTaskButton.className = 'add-task-button';
      addTaskButton.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M14 10H3v2h11v-2zm0-4H3v2h11V6zm4 8v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zM3 16h7v-2H3v2z"/></svg><span>Create Task</span>`;
      addTaskButton.onclick = () => openTaskForm(undefined, {
          sourceConversationId: state.activeConversationId,
          sourceMessageText: text,
          autoGenerateTitle: true
      });
      actions.appendChild(addTaskButton);
      messageGroup.appendChild(actions);

      if (groundingChunks && groundingChunks.length > 0) {
          const sourcesElement = createSourcesElement(groundingChunks);
          details.appendChild(sourcesElement);
      }
  }
  
  messageGroup.appendChild(avatar);
  messageGroup.appendChild(details);
  chatHistoryElement.appendChild(messageGroup);
  
  addCopyButtonsToCodeBlocks(messageElement);
  scrollToBottom();
  return { messageGroup, messageElement };
}

function scrollToBottom() {
  if(chatHistoryElement) {
    chatHistoryElement.scrollTop = chatHistoryElement.scrollHeight;
  }
}

// --- CORE CHAT LOGIC ---

async function handleFormSubmit(event: Event) {
  event.preventDefault();
  const userMessage = chatInput.value.trim();
  if (!userMessage || !ai) return;

  let currentConvoId = state.activeConversationId;

  // If no active chat, create a new one
  if (!currentConvoId) {
    currentConvoId = startNewChat(false); // don't re-render yet
  }

  const activeConvo = state.conversations[currentConvoId];
  const isFirstMessage = activeConvo.history.length === 0;

  appendMessage('user', userMessage);
  activeConvo.history.push({ role: 'user', text: userMessage });

  chatInput.value = '';
  chatInput.disabled = true;
  sendButton.disabled = true;
  
  showTypingIndicator();

  // If this was the first message, switch from welcome screen to chat view
  if (isFirstMessage) {
      welcomeScreen.style.display = 'none';
      chatHistoryElement.style.display = 'flex';
      setupConversation();
  }
  
  saveState();

  try {
    if (!chat) await setupConversation();
    if (!chat) throw new Error("Chat session not available.");
    
    const isWebSearchEnabled = activeTools.has('web-search');
    const config: GenerateContentConfig = {};
    if (isWebSearchEnabled) {
        config.tools = [{googleSearch: {}}];
    }

    const stream = await chat.sendMessageStream({ message: userMessage, config });
    hideTypingIndicator();
    
    let modelResponse = '';
    let groundingChunks: GroundingChunk[] = [];
    const { messageGroup, messageElement } = appendMessage('model', '');

    for await (const chunk of stream) {
      const chunkText = chunk.text;
      modelResponse += chunkText;
      const unsafeHtml = marked.parse(modelResponse + '▌') as string;
      messageElement.innerHTML = DOMPurify.sanitize(unsafeHtml);

      if (chunk.candidates?.[0]?.groundingMetadata?.groundingChunks) {
          groundingChunks = chunk.candidates[0].groundingMetadata.groundingChunks;
      }
      scrollToBottom();
    }
    
    const finalUnsafeHtml = marked.parse(modelResponse) as string;
    messageElement.innerHTML = DOMPurify.sanitize(finalUnsafeHtml);
    addCopyButtonsToCodeBlocks(messageElement);

    if (groundingChunks.length > 0) {
        const sourcesElement = createSourcesElement(groundingChunks);
        messageGroup.querySelector('.details').appendChild(sourcesElement);
    }
    
    activeConvo.history.push({ role: 'model', text: modelResponse, groundingChunks });
    
    if (isFirstMessage) {
        generateConversationTitle(currentConvoId);
    }
    
    saveState();
    
    // Non-blocking call to check for task suggestions
    checkForTaskSuggestion(activeConvo, messageGroup);


  } catch (error) {
    console.error("Error sending message:", error);
    appendMessage('model', 'Sorry, I encountered an error. Please try again.');
  } finally {
    hideTypingIndicator();
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
  }
}

async function generateConversationTitle(convoId: string) {
    const convo = state.conversations[convoId];
    if (!convo || convo.history.length < 2) return;

    const prompt = `Summarize the following conversation with a short, 3-5 word title.
    ---
    User: ${convo.history[0].text}
    Model: ${convo.history[1].text}
    ---
    Title:`;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });
        const title = response.text.trim().replace(/"/g, '');
        convo.title = title;
        chatTitle.textContent = title;
        renderConversationList();
        saveState();
    } catch (error) {
        console.error('Failed to generate title:', error);
    }
}

// --- CONVERSATION MANAGEMENT ---

function startNewChat(shouldRender = true): string {
  const newId = `convo_${Date.now()}`;
  const newConversation: Conversation = {
    id: newId,
    title: 'New Chat',
    history: [],
    customInstructions: state.settings.customInstructions,
    createdAt: Date.now(),
  };
  state.conversations[newId] = newConversation;
  state.activeConversationId = newId;
  saveState();

  if (shouldRender) {
    renderApp();
  }
  
  return newId;
}

function selectConversation(id: string) {
    if (state.activeConversationId === id) return;
    state.activeConversationId = id;
    saveState();
    renderApp();
    if (window.innerWidth < 768 && !state.settings.isSidebarPinned) {
        appContainer.classList.remove('sidebar-open');
    }
}

function deleteConversation(id: string, isArchived: boolean) {
    if (!confirm('Are you sure you want to permanently delete this conversation?')) return;
    
    const collection = isArchived ? state.archivedConversations : state.conversations;
    delete collection[id];

    if (state.activeConversationId === id) {
        const remainingIds = Object.keys(state.conversations);
        state.activeConversationId = remainingIds.length > 0 ? remainingIds[0] : null;
    }
    
    saveState();
    renderApp();
}

function toggleArchiveConversation(id: string) {
    const isArchived = id in state.archivedConversations;
    if (isArchived) {
        const convo = state.archivedConversations[id];
        delete state.archivedConversations[id];
        state.conversations[id] = convo;
        if (!state.activeConversationId) {
            state.activeConversationId = id;
        }
    } else {
        const convo = state.conversations[id];
        delete state.conversations[id];
        state.archivedConversations[id] = convo;
        if (state.activeConversationId === id) {
             const remainingIds = Object.keys(state.conversations);
             state.activeConversationId = remainingIds.length > 0 ? remainingIds[0] : null;
        }
    }
    saveState();
    renderApp();
}


// --- SIDEBAR ---

function toggleSidebar() {
    appContainer.classList.toggle('sidebar-open');
}

function updateSidebarPinnedState(isPinned: boolean) {
    state.settings.isSidebarPinned = isPinned;
    appContainer.classList.toggle('sidebar-pinned', isPinned);
    pinSidebarButton.classList.toggle('active', isPinned);
    pinSidebarButton.setAttribute('aria-label', isPinned ? 'Unpin sidebar' : 'Pin sidebar');
    saveState();
}

function handleSidebarMouseLeave() {
    if (!state.settings.isSidebarPinned) {
        appContainer.classList.remove('sidebar-open');
    }
}

// --- SETTINGS ---
function openSettings() {
    nicknameInput.value = state.settings.nickname;
    instructionsInput.value = state.settings.customInstructions;
    agentNameInput.value = state.settings.agentName;
    agentAvatarInput.value = state.settings.agentAvatarUrl;
    searchArchivedCheckbox.checked = state.settings.searchArchived;
    settingsOverlay.classList.remove('hidden');
}

function closeSettings() {
    settingsOverlay.classList.add('hidden');
}

function saveSettings() {
    state.settings.nickname = nicknameInput.value || 'You';
    state.settings.customInstructions = instructionsInput.value;
    state.settings.agentName = agentNameInput.value || DEFAULT_AGENT_NAME;
    state.settings.agentAvatarUrl = agentAvatarInput.value;
    state.settings.searchArchived = searchArchivedCheckbox.checked;
    saveState();
    closeSettings();
    renderApp();
    showNotification('Settings saved!', 'success');
}

// --- TOOL MANAGEMENT ---
function toggleTool(toolId: string) {
    if (activeTools.has(toolId)) {
        activeTools.delete(toolId);
    } else {
        activeTools.add(toolId);
    }
    renderActiveToolChips();
    renderToolSelector();
}

function renderToolSelector() {
    toolSelectorContainer.innerHTML = '';
    availableTools.forEach(tool => {
        const button = document.createElement('button');
        button.className = 'tool-button';
        button.dataset.toolId = tool.id;
        button.innerHTML = `${tool.icon}<span>${tool.label}</span>`;
        if (activeTools.has(tool.id)) {
            button.classList.add('active');
        }
        button.onclick = () => toggleTool(tool.id);
        toolSelectorContainer.appendChild(button);
    });
}

function renderActiveToolChips() {
    toolChipsContainer.innerHTML = '';
    activeTools.forEach(toolId => {
        const tool = availableTools.find(t => t.id === toolId);
        if (tool) {
            const chip = document.createElement('div');
            chip.className = 'tool-chip';
            chip.dataset.toolId = tool.id;

            const label = document.createElement('span');
            label.textContent = tool.label;

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-chip-button';
            removeBtn.innerHTML = '&times;';
            removeBtn.setAttribute('aria-label', `Deactivate ${tool.label}`);
            removeBtn.onclick = () => toggleTool(tool.id);

            chip.appendChild(label);
            chip.appendChild(removeBtn);
            toolChipsContainer.appendChild(chip);
        }
    });
}


// --- TASK MANAGEMENT ---

function openTasksOverlay() {
  renderTasks();
  tasksOverlay.classList.remove('hidden');
}

function closeTasksOverlay() {
  tasksOverlay.classList.add('hidden');
  closeTaskForm(); // Ensure form is also closed
}

function parseAndRenderSubtasks(task: Task): { html: string; completed: number; total: number } {
    const lines = task.notes.split('\n');
    const subtaskItems = [];
    let completed = 0;
    let total = 0;
    let subtaskIndex = 0;

    for (const line of lines) {
        const trimmedLine = line.trim();
        const match = trimmedLine.match(/^- \[( |x)\] (.*)/);
        if (match) {
            total++;
            const isChecked = match[1] === 'x';
            const text = match[2];
            if (isChecked) completed++;
            
            subtaskItems.push(`
                <li class="subtask-item ${isChecked ? 'completed' : ''}">
                    <input type="checkbox" id="subtask-${task.id}-${subtaskIndex}" class="subtask-checkbox" data-task-id="${task.id}" data-subtask-index="${subtaskIndex}" ${isChecked ? 'checked' : ''}>
                    <label for="subtask-${task.id}-${subtaskIndex}">${DOMPurify.sanitize(text)}</label>
                </li>
            `);
            subtaskIndex++;
        }
    }

    if (total === 0) {
        return { html: '', completed: 0, total: 0 };
    }

    const html = `
        <div class="subtask-container">
            <div class="subtask-progress">${completed} of ${total} completed</div>
            <ul class="subtask-list">${subtaskItems.join('')}</ul>
        </div>
    `;

    return { html, completed, total };
}

function renderTasks() {
  tasksList.innerHTML = '';
  const sortedTasks = [...state.tasks].sort((a, b) => {
      if (a.status === 'done' && b.status !== 'done') return 1;
      if (a.status !== 'done' && b.status === 'done') return -1;
      return b.createdAt - a.createdAt;
  });

  if (sortedTasks.length === 0) {
      tasksList.innerHTML = `<li class="no-tasks-message">You have no tasks yet. Create one from a conversation!</li>`;
      return;
  }

  sortedTasks.forEach(task => {
    const li = document.createElement('li');
    li.className = 'task-item';
    li.dataset.taskId = task.id;
    li.dataset.status = task.status;
    li.dataset.priority = task.priority;
    li.onclick = (e) => {
        // Prevent opening task form if a subtask checkbox/label was clicked
        const target = e.target as HTMLElement;
        if (target.closest('.subtask-item')) return;
        openTaskForm(task);
    };

    const dueDate = task.dueDate 
      ? new Date(task.dueDate).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' })
      : '';
    
    const { html: subtasksHtml } = parseAndRenderSubtasks(task);
      
    li.innerHTML = `
      <div class="task-item-content">
        <div class="task-item-header">
          <p class="task-item-title">${DOMPurify.sanitize(task.title)}</p>
          <span class="task-item-due-date">${dueDate}</span>
        </div>
        ${subtasksHtml}
        <div class="task-item-footer">
          <span class="task-status-pill" data-status="${task.status}">${task.status.replace('-', ' ')}</span>
        </div>
      </div>
    `;
    tasksList.appendChild(li);
  });
}

async function generateTitleForTask(fullText: string) {
    if (!ai) {
        taskTitleInput.value = fullText.substring(0, 50) + (fullText.length > 50 ? '...' : '');
        taskTitleInput.disabled = false;
        return;
    }
    try {
        const prompt = `Summarize the following text into a concise, actionable task title (under 10 words):\n\n---\n${fullText}\n---\n\nTitle:`;
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });
        const title = response.text.trim().replace(/"/g, '');
        taskTitleInput.value = title;
    } catch(e) {
        console.error("Failed to generate task title:", e);
        // Fallback to first line
        taskTitleInput.value = fullText.split('\n')[0].substring(0, 70);
    } finally {
        taskTitleInput.disabled = false;
        generateSubtasksButton.disabled = !taskTitleInput.value.trim();
    }
}

function openTaskForm(task?: Task, source?: { sourceConversationId: string, sourceMessageText: string, suggestedNotes?: string, autoGenerateTitle?: boolean }) {
  taskForm.reset();
  editingTaskId = task?.id || null;

  if (task) {
    taskFormTitle.textContent = 'Edit Task';
    taskIdInput.value = task.id;
    taskTitleInput.value = task.title;
    taskNotesInput.value = task.notes;
    taskStatusSelect.value = task.status;
    taskPrioritySelect.value = task.priority;
    taskDueDateInput.value = task.dueDate || '';
    taskDeleteButton.classList.remove('hidden');
    
    if (task.sourceConversationId) {
        taskFormSourceLink.classList.remove('hidden');
        taskFormSourceLink.onclick = (e) => {
            e.preventDefault();
            selectConversation(task.sourceConversationId);
            closeTasksOverlay();
        };
    } else {
        taskFormSourceLink.classList.add('hidden');
    }

  } else {
    taskFormTitle.textContent = 'New Task';
    editingTaskId = null;
    taskIdInput.value = '';
    taskDeleteButton.classList.add('hidden');
    taskFormSourceLink.classList.add('hidden');

    if (source?.autoGenerateTitle) {
        taskTitleInput.value = 'Generating title...';
        taskTitleInput.disabled = true;
        taskNotesInput.value = source.sourceMessageText; // Full text goes to notes
        generateTitleForTask(source.sourceMessageText);
    } else {
        // From AI suggestion or manual creation
        taskTitleInput.value = source?.sourceMessageText || '';
        taskNotesInput.value = source?.suggestedNotes || '';
    }
  }
  
  generateSubtasksButton.disabled = !taskTitleInput.value.trim() || taskTitleInput.disabled;
  taskFormModal.dataset.sourceConvoId = source?.sourceConversationId || task?.sourceConversationId || '';
  taskFormModal.dataset.sourceMessage = source?.sourceMessageText || task?.sourceMessageText || '';
  taskFormModal.classList.remove('hidden');
}

function closeTaskForm() {
  editingTaskId = null;
  taskFormModal.classList.add('hidden');
  taskForm.reset();
}

function handleTaskFormSubmit(e: Event) {
  e.preventDefault();
  const formData = {
    id: taskIdInput.value,
    title: taskTitleInput.value.trim(),
    notes: taskNotesInput.value.trim(),
    status: taskStatusSelect.value as TaskStatus,
    priority: taskPrioritySelect.value as TaskPriority,
    dueDate: taskDueDateInput.value || null,
  };

  if (!formData.title) {
    showNotification('Task title cannot be empty', 'error');
    return;
  }

  if (editingTaskId) {
    const taskIndex = state.tasks.findIndex(t => t.id === editingTaskId);
    if (taskIndex > -1) {
      const existingTask = state.tasks[taskIndex];
      state.tasks[taskIndex] = { ...existingTask, ...formData };
      showNotification('Task updated', 'success');
    }
  } else {
    const newTask: Task = {
      ...formData,
      id: `task_${Date.now()}`,
      createdAt: Date.now(),
      sourceConversationId: taskFormModal.dataset.sourceConvoId || null,
      sourceMessageText: taskFormModal.dataset.sourceMessage || null,
    };
    state.tasks.push(newTask);
    showNotification('Task created', 'success');
  }

  saveState();
  renderTasks();
  closeTaskForm();
}

function handleDeleteTask() {
    if (!editingTaskId) return;
    if (confirm('Are you sure you want to delete this task?')) {
        state.tasks = state.tasks.filter(t => t.id !== editingTaskId);
        saveState();
        renderTasks();
        closeTaskForm();
        showNotification('Task deleted', 'info');
    }
}

async function handleGenerateSubtasks() {
    const title = taskTitleInput.value.trim();
    const notes = taskNotesInput.value.trim();
    if (!title || !ai) return;

    generateSubtasksButton.disabled = true;
    generateSubtasksButton.classList.add('loading');

    try {
        const prompt = `Based on the task title "${title}" and any existing notes "${notes}", generate a checklist of actionable sub-tasks to complete it. Return a JSON array of strings, where each string is a sub-task.`;
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: { type: Type.ARRAY, items: { type: Type.STRING } }
            },
        });
        const subtasks = JSON.parse(response.text) as string[];
        if (subtasks && subtasks.length > 0) {
            const checklist = subtasks.map(task => `- [ ] ${task}`).join('\n');
            const separator = taskNotesInput.value.trim() ? '\n\n' : '';
            taskNotesInput.value += `${separator}${checklist}`;
            showNotification('Sub-tasks generated!', 'success');
        } else {
            showNotification('Could not generate sub-tasks.', 'info');
        }
    } catch (error) {
        console.error("Failed to generate sub-tasks:", error);
        showNotification('Error generating sub-tasks.', 'error');
    } finally {
        generateSubtasksButton.classList.remove('loading');
        generateSubtasksButton.disabled = false;
    }
}

async function checkForTaskSuggestion(conversation: Conversation, messageGroup: HTMLElement) {
    if (!ai || conversation.history.length < 2) return;

    try {
        const lastUserEntry = conversation.history[conversation.history.length - 2];
        const lastModelEntry = conversation.history[conversation.history.length - 1];

        const prompt = `Analyze the following exchange. Does it imply a clear, actionable task for the user?
        User: "${lastUserEntry.text}"
        Assistant: "${lastModelEntry.text}"
        
        Respond in JSON. If a task is implied, set "taskSuggested" to true and provide a concise "taskTitle" and detailed "taskNotes". If not, set "taskSuggested" to false.`;
        
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        taskSuggested: { type: Type.BOOLEAN },
                        taskTitle: { type: Type.STRING },
                        taskNotes: { type: Type.STRING }
                    },
                }
            }
        });

        const suggestion = JSON.parse(response.text) as TaskSuggestion;
        if (suggestion.taskSuggested) {
            renderTaskSuggestion(messageGroup, suggestion, conversation.id, lastModelEntry.text);
        }
    } catch (error) {
        // Fail silently as this is a non-critical background task
        console.error("Error checking for task suggestion:", error);
    }
}

function renderTaskSuggestion(messageGroup: HTMLElement, suggestion: TaskSuggestion, convoId: string, messageText: string) {
    const detailsContainer = messageGroup.querySelector('.details');
    if (!detailsContainer) return;

    const bar = document.createElement('div');
    bar.className = 'task-suggestion-bar';

    bar.innerHTML = `
        <div class="task-suggestion-content">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M4 19h16v-2H4v2zm16-4h-2v-4h-4V9h4V5h2v4h4v2h-4v4zM2 13h11v-2H2v2zm0-4h11V7H2v2zm0-4h11V3H2v2z"/></svg>
            <span class="task-suggestion-text"><strong>AI Suggestion:</strong> ${DOMPurify.sanitize(suggestion.taskTitle)}</span>
        </div>
        <div class="task-suggestion-actions">
            <button class="dismiss-suggestion">Dismiss</button>
            <button class="create-task-suggestion primary">Create Task</button>
        </div>
    `;

    bar.querySelector('.create-task-suggestion').addEventListener('click', () => {
        openTasksOverlay();
        openTaskForm(undefined, {
            sourceConversationId: convoId,
            sourceMessageText: suggestion.taskTitle, // Use the more concise title from AI
            suggestedNotes: suggestion.taskNotes
        });
        bar.remove();
    });

    bar.querySelector('.dismiss-suggestion').addEventListener('click', () => {
        bar.remove();
    });

    detailsContainer.appendChild(bar);
}



// --- UTILITIES & HELPERS ---

function showNotification(message: string, type: 'success' | 'error' | 'info' = 'info', duration = 3000) {
    if (!notificationContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast-notification ${type}`;
    toast.textContent = message;

    notificationContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fade-out');
        toast.addEventListener('animationend', () => {
            toast.remove();
        });
    }, duration);
}

function showTypingIndicator() {
  if (document.getElementById(TYPING_INDICATOR_ID)) return;
  const indicatorGroup = document.createElement('div');
  indicatorGroup.id = TYPING_INDICATOR_ID;
  indicatorGroup.classList.add('message-group', 'model');
  
  const avatarHtml = state.settings.agentAvatarUrl
    ? `<img src="${state.settings.agentAvatarUrl}" alt="Agent Avatar" onerror="this.outerHTML='${AGENT_AVATAR}'">`
    : AGENT_AVATAR;

  indicatorGroup.innerHTML = `
    <div class="avatar">${avatarHtml}</div>
    <div class="details">
      <div class="name">${state.settings.agentName}</div>
      <div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
    </div>`;
  chatHistoryElement.appendChild(indicatorGroup);
  scrollToBottom();
}

function hideTypingIndicator() {
  const indicator = document.getElementById(TYPING_INDICATOR_ID);
  if (indicator) indicator.remove();
}

function addCopyButtonsToCodeBlocks(container: HTMLElement) {
    const codeBlocks = container.querySelectorAll('pre');
    codeBlocks.forEach(pre => {
        if (pre.querySelector('.copy-code-button')) return;
        const code = pre.querySelector('code');
        if (!code) return;
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-code-button';
        copyButton.textContent = 'Copy';
        copyButton.setAttribute('aria-label', 'Copy code to clipboard');
        pre.appendChild(copyButton);
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(code.innerText).then(() => {
                copyButton.textContent = 'Copied!';
                showNotification('Code copied to clipboard', 'success');
                setTimeout(() => { copyButton.textContent = 'Copy'; }, 2000);
            }).catch(err => { 
                console.error('Failed to copy code: ', err);
                showNotification('Failed to copy code', 'error');
            });
        });
    });
}

async function generateAndDisplaySamplePrompts() {
  if (!ai || !samplePromptsContainer) return;
  samplePromptsContainer.innerHTML = '<div class="spinner"></div>';

  try {
    const prompt = `Generate 4 concise and engaging sample prompts for a user to ask a general-purpose AI assistant. Focus on creative, educational, or practical topics. Return the prompts as a JSON array of strings.`;
    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: { responseMimeType: "application/json", responseSchema: { type: Type.ARRAY, items: { type: Type.STRING } } },
    });
    const prompts = JSON.parse(response.text);
    renderSamplePrompts(prompts);
  } catch (error) {
    console.error("Failed to generate sample prompts:", error);
    renderSamplePrompts(STATIC_SAMPLE_PROMPTS); // Fallback
  }
}

function renderSamplePrompts(prompts: string[]) {
    if (!samplePromptsContainer) return;
    samplePromptsContainer.innerHTML = '';
    prompts.forEach(promptText => {
        const button = document.createElement('button');
        button.classList.add('sample-prompt');
        button.textContent = promptText;
        button.addEventListener('click', () => {
            chatInput.value = promptText;
            chatForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
        });
        samplePromptsContainer.appendChild(button);
    });
}

// --- COMMAND PALETTE ---
function renderCommands(filter = '') {
    if (!commandPaletteList) return;
    const lowerCaseFilter = filter.toLowerCase();
    const filteredCommands = commands.filter(cmd => cmd.label.toLowerCase().includes(lowerCaseFilter) || cmd.keywords?.some(k => k.toLowerCase().includes(lowerCaseFilter)));
    commandPaletteList.innerHTML = '';
    filteredCommands.forEach((cmd, index) => {
        const li = document.createElement('li');
        li.dataset.commandId = cmd.id;
        li.innerHTML = `<span class="command-label">${cmd.label}</span>`;
        if (index === activeCommandIndex) li.classList.add('selected');
        li.addEventListener('click', () => { cmd.action(); closeCommandPalette(); });
        commandPaletteList.appendChild(li);
    });
    if (filteredCommands.length > 0 && activeCommandIndex >= filteredCommands.length) {
        activeCommandIndex = filteredCommands.length - 1;
        updateCommandSelection();
    } else if (filteredCommands.length === 0) {
        activeCommandIndex = -1;
    }
}
function updateCommandSelection() {
    const items = commandPaletteList.querySelectorAll('li');
    items.forEach((item, index) => {
        item.classList.toggle('selected', index === activeCommandIndex);
        if (index === activeCommandIndex) item.scrollIntoView({ block: 'nearest' });
    });
}
function openCommandPalette() {
    commandPaletteOverlay.classList.remove('hidden');
    commandPaletteInput.focus();
    activeCommandIndex = 0;
    renderCommands();
}
function closeCommandPalette() {
    commandPaletteOverlay.classList.add('hidden');
    commandPaletteInput.value = '';
}
function handleCommandPaletteNavigation(e: KeyboardEvent) {
    const items = commandPaletteList.querySelectorAll('li');
    if (!items.length) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); activeCommandIndex = (activeCommandIndex + 1) % items.length; updateCommandSelection(); } 
    else if (e.key === 'ArrowUp') { e.preventDefault(); activeCommandIndex = (activeCommandIndex - 1 + items.length) % items.length; updateCommandSelection(); } 
    else if (e.key === 'Enter') { e.preventDefault();
        const selectedItem = items[activeCommandIndex] as HTMLLIElement;
        const command = commands.find(c => c.id === selectedItem.dataset.commandId);
        if (command) { command.action(); closeCommandPalette(); }
    }
}

// --- MAIN EXECUTION ---
async function main() {
  const savedTheme = localStorage.getItem('gemini-theme') as 'light' | 'dark' | null;
  applyTheme(savedTheme || 'dark');
  
  loadState();
  initializeChat();
  renderApp();
  renderToolSelector();

  // --- Event Listeners ---
  chatForm.addEventListener('submit', handleFormSubmit);
  themeSwitcher.addEventListener('click', toggleTheme);
  newChatButton.addEventListener('click', () => startNewChat());
  sidebarToggleButton.addEventListener('click', toggleSidebar);
  pinSidebarButton.addEventListener('click', () => updateSidebarPinnedState(!state.settings.isSidebarPinned));
  conversationSidebar.addEventListener('mouseleave', handleSidebarMouseLeave);
  searchInput.addEventListener('input', renderConversationList);
  settingsButton.addEventListener('click', openSettings);
  settingsSaveButton.addEventListener('click', saveSettings);
  settingsCloseButton.addEventListener('click', closeSettings);
  settingsOverlay.addEventListener('click', (e) => { if (e.target === settingsOverlay) closeSettings(); });
  
  // Task Listeners
  tasksButton.addEventListener('click', openTasksOverlay);
  closeTasksButton.addEventListener('click', closeTasksOverlay);
  tasksOverlay.addEventListener('click', (e) => { if (e.target === tasksOverlay) closeTasksOverlay(); });
  createNewTaskButton.addEventListener('click', () => openTaskForm());
  taskForm.addEventListener('submit', handleTaskFormSubmit);
  taskCancelButton.addEventListener('click', closeTaskForm);
  taskDeleteButton.addEventListener('click', handleDeleteTask);
  generateSubtasksButton.addEventListener('click', handleGenerateSubtasks);
  taskTitleInput.addEventListener('keyup', () => {
    generateSubtasksButton.disabled = !taskTitleInput.value.trim();
  });
  tasksList.addEventListener('change', (e) => {
    const target = e.target as HTMLInputElement;
    if (target.classList.contains('subtask-checkbox')) {
        const taskId = target.dataset.taskId;
        const subtaskIndex = parseInt(target.dataset.subtaskIndex, 10);
        const task = state.tasks.find(t => t.id === taskId);
        if (!task) return;

        const lines = task.notes.split('\n');
        let currentSubtask = -1;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.trim().startsWith('- [')) {
                currentSubtask++;
                if (currentSubtask === subtaskIndex) {
                    lines[i] = target.checked
                        ? line.replace('[ ]', '[x]')
                        : line.replace('[x]', '[ ]');
                    break;
                }
            }
        }
        task.notes = lines.join('\n');
        saveState();
        renderTasks(); // Re-render to update progress and styles
    }
  });


  // Command Palette Listeners
  commandPaletteInput.addEventListener('input', () => { activeCommandIndex = 0; renderCommands(commandPaletteInput.value); });
  commandPaletteOverlay.addEventListener('click', (e) => { if (e.target === commandPaletteOverlay) closeCommandPalette(); });
  commandPaletteInput.addEventListener('keydown', handleCommandPaletteNavigation);
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); openCommandPalette(); }
    if (e.key === 'Escape') {
      if (!tasksOverlay.classList.contains('hidden')) {
          if (!taskFormModal.classList.contains('hidden')) {
              closeTaskForm();
          } else {
              closeTasksOverlay();
          }
      }
      if (!commandPaletteOverlay.classList.contains('hidden')) closeCommandPalette();
      if (!settingsOverlay.classList.contains('hidden')) closeSettings();
    }
  });
  
  // Scrollbar visibility logic
  if (chatHistoryElement) {
    let scrollTimeout: number;
    chatHistoryElement.addEventListener('scroll', () => {
      chatHistoryElement.classList.add('is-scrolling');
      clearTimeout(scrollTimeout);
      scrollTimeout = window.setTimeout(() => {
        chatHistoryElement.classList.remove('is-scrolling');
      }, 1000);
    }, { passive: true });
  }
}

main();