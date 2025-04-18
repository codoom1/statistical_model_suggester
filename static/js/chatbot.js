/**
 * Chatbot functionality for the Statistical Model Suggester application
 */

class ChatBot {
    constructor() {
        this.chatIcon = document.getElementById('chat-icon');
        this.chatWindow = document.getElementById('chat-window');
        this.messageContainer = document.getElementById('chat-messages');
        this.userInput = document.getElementById('user-message');
        this.sendButton = document.getElementById('send-message');
        this.closeButton = document.getElementById('close-chat');
        this.pageContext = document.querySelector('meta[name="page-context"]')?.content || '';
        
        this.setupEventListeners();
        this.addWelcomeMessage();
    }
    
    setupEventListeners() {
        // Toggle chat window
        this.chatIcon.addEventListener('click', () => this.toggleChatWindow());
        this.closeButton.addEventListener('click', () => this.toggleChatWindow(false));
        
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }
    
    toggleChatWindow(show) {
        const shouldShow = show !== undefined ? show : !this.chatWindow.classList.contains('show');
        
        if (shouldShow) {
            this.chatWindow.classList.add('show');
            this.userInput.focus();
        } else {
            this.chatWindow.classList.remove('show');
        }
    }
    
    addWelcomeMessage() {
        const welcomeMessage = "👋 Hi there! I'm your AI assistant. How can I help you with Statistical Model Suggester today?";
        this.addMessage(welcomeMessage, 'bot');
    }
    
    addMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}-message`;
        
        // Create message bubble
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = message;
        
        messageElement.appendChild(bubble);
        this.messageContainer.appendChild(messageElement);
        
        // Scroll to bottom
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    addTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.className = 'message bot-message typing-indicator';
        typingElement.id = 'typing-indicator';
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = '<span></span><span></span><span></span>';
        
        typingElement.appendChild(bubble);
        this.messageContainer.appendChild(typingElement);
        
        // Scroll to bottom
        this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
    
    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    async sendMessage() {
        const userMessage = this.userInput.value.trim();
        
        if (!userMessage) {
            return;
        }
        
        // Add user message to chat
        this.addMessage(userMessage, 'user');
        
        // Clear input
        this.userInput.value = '';
        
        // Show typing indicator
        this.addTypingIndicator();
        
        try {
            // Call API to get response
            const response = await fetch('/chatbot/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: userMessage,
                    context: this.pageContext
                }),
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            if (data.success) {
                // Add bot response
                this.addMessage(data.response, 'bot');
            } else {
                // Check for specific error types
                if (response.status === 402 || data.message === 'API credits exceeded') {
                    // Add a more helpful error message for credit limits
                    this.addMessage("I apologize, but our AI service has reached its usage limit for this month. Basic functionality will continue to work, but AI-powered features may be limited until the next billing cycle.", 'bot');
                    
                    // Add a follow-up message with alternatives
                    setTimeout(() => {
                        this.addMessage("In the meantime, you can still use all the non-AI features of the application. If you have specific questions about statistical models, you might find helpful information in the models section.", 'bot');
                    }, 1000);
                } else {
                    // Add general error message
                    this.addMessage(data.response || 'Sorry, I encountered an error. Please try again later.', 'bot');
                }
            }
        } catch (error) {
            console.error('Error sending message:', error);
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            // Add error message
            this.addMessage('Sorry, I encountered an error. Please try again later.', 'bot');
        }
    }
}

// Initialize chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if chat elements exist before initializing
    if (document.getElementById('chat-icon')) {
        window.chatBot = new ChatBot();
    }
}); 