import gradio as gr

# Question bank
QUESTIONS = [
    "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "3) Tell me about a time when you made a mistake at work. How did you address it?",
    "4) How do you handle situations where you need to learn something new?",
    "5) Can you share an example of when you had to adapt to a significant change at work?"
]

def analyze_answers(*answers):
    """Analyze the answers and generate scores."""
    scores = []
    for answer in answers:
        # Placeholder for analysis logic
        # Here you can implement your scoring logic based on the answer
        score = len(answer) % 10  # Example scoring logic based on answer length
        scores.append(score)

    return scores

def generate_report(*answers):
    """Generate a report based on the answers."""
    scores = analyze_answers(*answers)
    report = "### Candidate Report\n\n"
    
    for i, (question, answer, score) in enumerate(zip(QUESTIONS, answers, scores)):
        report += f"**Q: {question}**\n"
        report += f"A: {answer}\n"
        report += f"Score: {score}\n\n"

    return report

# Gradio UI
with gr.Blocks(title="Interview Report Generator") as demo:
    gr.Markdown("## AI-Powered Interview Report Generator")

    answer_inputs = []
    for question in QUESTIONS:
        answer_input = gr.Textbox(label=question, placeholder="Enter your answer here")
        answer_inputs.append(answer_input)

    generate_btn = gr.Button("Generate Report")
    report_display = gr.Markdown("")

    generate_btn.click(
        fn=generate_report,
        inputs=answer_inputs,
        outputs=report_display
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
