import streamlit as st
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List
import re

# Import your custom modules (make sure these files are in the same directory)
try:
    from classifier import detect_stress
    from extractor import extract_signals
    from generate_response import empathetic_reply
    from config import STRESS_THRESHOLD
    from evaluation import evaluate_classifier, evaluate_responses
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all required Python files (classifier.py, extractor.py, generate_response.py, config.py, evaluation.py) are in the same directory as this Streamlit app.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MindCare - Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}

.chat-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.user-message {
    background-color: #e3f2fd;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    border-left: 4px solid #2196F3;
}

.bot-message {
    background-color: #f1f8e9;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    border-left: 4px solid #4CAF50;
}

.stress-high {
    color: #f44336;
    font-weight: bold;
}

.stress-medium {
    color: #ff9800;
    font-weight: bold;
}

.stress-low {
    color: #4caf50;
    font-weight: bold;
}

.urgent-alert {
    background-color: #ffebee;
    border: 2px solid #f44336;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.sidebar-info {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
            color:black;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'stress_history' not in st.session_state:
    st.session_state.stress_history = []
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'accuracy_data' not in st.session_state:
    st.session_state.accuracy_data = {"predictions": [], "actual": [], "responses": [], "references": []}

def clean_bot_response(response_text: str) -> str:
    """Clean and format bot response to remove raw dictionary data"""
    if not isinstance(response_text, str):
        return str(response_text)
    
    # Remove dictionary representations and debugging info
    patterns_to_remove = [
        r"Symptoms: \[.*?\]",
        r"'triggers': \[.*?\]",
        r"'symptoms': \[.*?\]", 
        r"'coping': \[.*?\]",
        r"'red_flags': \[.*?\]",
        r"'urgent': \w+",
        r"\{.*?\}",  # Remove any remaining dictionary representations
    ]
    
    cleaned = response_text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
    
    # Clean up extra whitespace and line breaks
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def calculate_model_accuracy():
    """Calculate accuracy metrics for the chatbot"""
    if len(st.session_state.accuracy_data["predictions"]) < 2:
        return None

def create_accuracy_dashboard():
    """Create accuracy metrics dashboard"""
    accuracy_metrics = calculate_model_accuracy()
    
    if accuracy_metrics is None:
        st.info("📊 Accuracy metrics will be available after more interactions")
        return
    
    st.subheader("🎯 Model Performance")
    
    # Classifier metrics
    classifier_metrics = accuracy_metrics["classifier"]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{classifier_metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{classifier_metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{classifier_metrics['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{classifier_metrics['f1']:.3f}")
    
    # Response quality metrics (if available)
    if accuracy_metrics["responses"]:
        st.subheader("📝 Response Quality")
        response_metrics = accuracy_metrics["responses"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BLEU Score", f"{response_metrics['bleu']:.3f}")
        with col2:
            st.metric("ROUGE-1", f"{response_metrics['rouge1']:.3f}")
        with col3:
            st.metric("ROUGE-L", f"{response_metrics['rougeL']:.3f}")
    
    # Accuracy over time chart
    if len(st.session_state.accuracy_data["predictions"]) > 1:
        accuracy_over_time = []
        correct_predictions = 0
        
        for i, (pred, actual) in enumerate(zip(
            st.session_state.accuracy_data["predictions"],
            st.session_state.accuracy_data["actual"]
        )):
            if pred == actual:
                correct_predictions += 1
            accuracy_over_time.append(correct_predictions / (i + 1))
        
        if accuracy_over_time:
            fig = px.line(
                x=list(range(1, len(accuracy_over_time) + 1)),
                y=accuracy_over_time,
                title="Classification Accuracy Over Time",
                labels={'x': 'Interaction Number', 'y': 'Cumulative Accuracy'},
                color_discrete_sequence=['#2196F3']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    try:
        # Stress classification accuracy
        y_true = st.session_state.accuracy_data["actual"]
        y_pred = st.session_state.accuracy_data["predictions"]
        
        # Convert stress labels to binary (high stress = 1, others = 0)
        y_true_binary = [1 if label == "high" else 0 for label in y_true]
        y_pred_binary = [1 if label == "high" else 0 for label in y_pred]
        
        classifier_metrics = evaluate_classifier(y_true_binary, y_pred_binary)
        
        # Response quality metrics (if we have reference responses)
        response_metrics = None
        if st.session_state.accuracy_data["references"]:
            responses = st.session_state.accuracy_data["responses"]
            references = st.session_state.accuracy_data["references"]
            if len(responses) > 0 and len(references) > 0:
                response_metrics = evaluate_responses(responses, references)
        
        return {
            "classifier": classifier_metrics,
            "responses": response_metrics,
            "total_interactions": len(y_true)
        }
    except Exception as e:
        st.error(f"Error calculating accuracy: {e}")
        return None
def get_stress_color_class(stress_label: str) -> str:
    """Return CSS class based on stress level"""
    if stress_label == "high":
        return "stress-high"
    elif stress_label == "medium":
        return "stress-medium"
    else:
        return "stress-low"

def format_chat_message(message: dict) -> str:
    """Format a chat message for display"""
    timestamp = message.get('timestamp', '')
    user_text = message.get('user_text', '')
    bot_response = message.get('bot_response', '')
    stress_info = message.get('stress_info', {})
    
    stress_label = stress_info.get('stress_label', 'unknown')
    stress_score = stress_info.get('stress_score', 0.0)
    stress_class = get_stress_color_class(stress_label)
    
    return f"""
    <div class="user-message">
        <strong>You ({timestamp}):</strong><br>
        {user_text}
        <br><small>Detected stress: <span class="{stress_class}">{stress_label.upper()}</span> ({stress_score:.2f})</small>
    </div>
    <div class="bot-message">
        <strong>MindCare Bot:</strong><br>
        {bot_response}
    </div>
    """

def create_stress_chart():
    """Create a chart showing stress levels over time"""
    if not st.session_state.stress_history:
        return None
    
    df = pd.DataFrame(st.session_state.stress_history)
    
    fig = px.line(df, x='timestamp', y='stress_score', 
                  title='Stress Level Over Time',
                  labels={'stress_score': 'Stress Score', 'timestamp': 'Time'},
                  color_discrete_sequence=['#2196F3'])
    
    # Add color zones
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="High Stress Zone")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Stress Zone")
    
    fig.update_layout(height=400)
    return fig

def create_emotion_chart():
    """Create a chart showing emotion distribution"""
    if not st.session_state.emotion_history:
        return None
    
    # Aggregate emotions across all messages
    all_emotions = {}
    for entry in st.session_state.emotion_history:
        for emotion, score in entry['emotions'].items():
            if emotion not in all_emotions:
                all_emotions[emotion] = []
            all_emotions[emotion].append(score)
    
    # Calculate average scores
    avg_emotions = {emotion: sum(scores)/len(scores) 
                   for emotion, scores in all_emotions.items()}
    
    if avg_emotions:
        fig = px.bar(x=list(avg_emotions.keys()), y=list(avg_emotions.values()),
                     title='Average Emotion Distribution',
                     labels={'x': 'Emotion', 'y': 'Average Score'},
                     color=list(avg_emotions.values()),
                     color_continuous_scale='viridis')
        fig.update_layout(height=400)
        return fig
    
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 MindCare - Mental Health Chatbot</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard")
        
        # User info
        st.subheader("User Information")
        user_name = st.text_input("Your Name (Optional)", 
                                  value=st.session_state.user_name,
                                  placeholder="Enter your name...")
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
        
        # Statistics
        st.subheader("Session Statistics")
        total_messages = len(st.session_state.chat_history)
        st.metric("Total Messages", total_messages)
        
        if st.session_state.stress_history:
            avg_stress = sum(entry['stress_score'] for entry in st.session_state.stress_history) / len(st.session_state.stress_history)
            st.metric("Average Stress Level", f"{avg_stress:.2f}")
            
            current_stress = st.session_state.stress_history[-1]['stress_label'] if st.session_state.stress_history else "N/A"
            st.metric("Current Stress Level", current_stress.capitalize())
        
        # Emergency contacts
        st.subheader("🆘 Emergency Resources")
        st.markdown("""
        <div class="sidebar-info">
        <strong>If you're in crisis:</strong><br>
        • National Suicide Prevention Lifeline: 988<br>
        • Crisis Text Line: Text HOME to 741741<br>
        • Emergency Services: 911<br>
        </div>
        """, unsafe_allow_html=True)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.stress_history = []
            st.session_state.emotion_history = []
            st.session_state.accuracy_data = {"predictions": [], "actual": [], "responses": [], "references": []}
            st.rerun()
        
        # Export data
        if st.session_state.chat_history:
            if st.button("📥 Export Chat Data", type="secondary"):
                export_data = {
                    'chat_history': st.session_state.chat_history,
                    'stress_history': st.session_state.stress_history,
                    'emotion_history': st.session_state.emotion_history,
                    'export_timestamp': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"mindcare_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Chat input
        user_input = st.text_area(
            "How are you feeling today? Share what's on your mind...",
            height=100,
            placeholder="I've been feeling really stressed about my exams coming up..."
        )
        
        # Model selection
        response_model = st.selectbox(
            "Choose Response Model:",
            ["Both (FLAN + Groq)", "FLAN-T5", "Groq (Llama)"],
            index=0
        )
        
        # Optional: User feedback for accuracy calculation
        if st.session_state.chat_history:
            with st.expander("💯 Rate Last Response (Optional - for accuracy calculation)"):
                st.write("How accurate was the stress detection in your last message?")
                actual_stress = st.selectbox(
                    "Your actual stress level was:",
                    ["Select...", "low", "medium", "high"],
                    key="stress_feedback"
                )
                
                if st.button("Submit Feedback") and actual_stress != "Select...":
                    last_prediction = st.session_state.stress_history[-1]['stress_label']
                    st.session_state.accuracy_data["predictions"].append(last_prediction)
                    st.session_state.accuracy_data["actual"].append(actual_stress)
                    st.success("Thank you for your feedback! This helps improve the model's accuracy.")
        
        # Process user input
        if st.button("💨 Send Message", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing your message and generating response..."):
                    try:
                        # Detect stress and emotions
                        stress_result = detect_stress(user_input)
                        
                        # Extract signals (triggers, symptoms, coping strategies)
                        signals = extract_signals(user_input)
                        
                        # Check for urgent situations
                        if signals.get('urgent', False):
                            st.error("⚠️ URGENT: Your message contains concerning content. Please reach out to emergency services or a trusted person immediately.")
                            st.markdown("""
                            <div class="urgent-alert">
                            <strong>🆘 Immediate Help Resources:</strong><br>
                            • Call 988 (Suicide & Crisis Lifeline)<br>
                            • Text "HELLO" to 741741 (Crisis Text Line)<br>
                            • Call 911 for emergency services<br>
                            • Go to your nearest emergency room<br>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Generate empathetic response
                        responses = empathetic_reply(
                            user_input, 
                            stress_result['stress_label'],
                            stress_result['stress_score'],
                            signals
                        )
                        
                        # Choose which response to show and clean it
                        if response_model == "FLAN-T5":
                            bot_response = clean_bot_response(responses['flan'])
                        elif response_model == "Groq (Llama)":
                            bot_response = clean_bot_response(responses['groq'])
                        else:
                            flan_clean = clean_bot_response(responses['flan'])
                            groq_clean = clean_bot_response(responses['groq'])
                            bot_response = f"**FLAN-T5 Response:**\n{flan_clean}\n\n**Groq Response:**\n{groq_clean}"
                        
                        # Store cleaned response for accuracy evaluation
                        st.session_state.accuracy_data["responses"].append(bot_response)
                        
                        # Store in chat history
                        chat_entry = {
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'user_text': user_input,
                            'bot_response': bot_response,
                            'stress_info': stress_result,
                            'signals': signals,
                            'model_used': response_model
                        }
                        st.session_state.chat_history.append(chat_entry)
                        
                        # Store stress and emotion history
                        st.session_state.stress_history.append({
                            'timestamp': datetime.now(),
                            'stress_label': stress_result['stress_label'],
                            'stress_score': stress_result['stress_score']
                        })
                        
                        st.session_state.emotion_history.append({
                            'timestamp': datetime.now(),
                            'emotions': stress_result.get('emotions', {})
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing your message: {str(e)}")
                        st.error("Please make sure all required dependencies are installed and API keys are configured.")
            else:
                st.warning("Please enter a message before sending.")
        
        # Display chat history
        st.subheader("📝 Conversation History")
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                for message in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
                    st.markdown(format_chat_message(message), unsafe_allow_html=True)
            else:
                st.info("👋 Welcome! Start a conversation by sharing how you're feeling today.")
    
    with col2:
        # Create tabs for different analytics
        tab1, tab2 = st.tabs(["📈 Analytics", "🎯 Accuracy"])
        
        with tab1:
            st.subheader("📈 Analytics")
            
            # Stress level chart
            if st.session_state.stress_history:
                stress_fig = create_stress_chart()
                if stress_fig:
                    st.plotly_chart(stress_fig, use_container_width=True)
            
            # Emotion distribution chart
            if st.session_state.emotion_history:
                emotion_fig = create_emotion_chart()
                if emotion_fig:
                    st.plotly_chart(emotion_fig, use_container_width=True)
            
            # Latest analysis
            if st.session_state.chat_history:
                st.subheader("🔍 Latest Analysis")
                latest = st.session_state.chat_history[-1]
                
                # Stress info
                stress_info = latest['stress_info']
                stress_class = get_stress_color_class(stress_info['stress_label'])
                st.markdown(f"""
                **Stress Level:** <span class="{stress_class}">{stress_info['stress_label'].upper()}</span>  
                **Stress Score:** {stress_info['stress_score']:.2f}
                """, unsafe_allow_html=True)
                
                # Signals
                signals = latest['signals']
                if signals:
                    if signals.get('symptoms'):
                        st.write("**Detected Symptoms:**", ", ".join(signals['symptoms']))
                    if signals.get('triggers'):
                        st.write("**Potential Triggers:**", ", ".join(signals['triggers']))
                    if signals.get('coping'):
                        st.write("**Coping Strategies Mentioned:**", ", ".join(signals['coping']))
                    if signals.get('red_flags'):
                        st.error("**⚠️ Red Flags Detected:**", ", ".join(signals['red_flags']))
                
                # Emotions
                emotions = stress_info.get('emotions', {})
                if emotions:
                    st.write("**Top Emotions:**")
                    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    for emotion, score in sorted_emotions:
                        st.write(f"• {emotion.capitalize()}: {score:.2f}")
        
        with tab2:
            create_accuracy_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
    <p>🧠 MindCare Mental Health Chatbot | Built with Streamlit</p>
    <p><strong>Disclaimer:</strong> This chatbot is for support purposes only and is not a replacement for professional mental health care. 
    If you're experiencing a mental health emergency, please contact emergency services immediately.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()