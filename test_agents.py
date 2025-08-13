# # test_agents.py
# import asyncio
# import sys
# import os

# # Add project root to Python path to allow imports from core, backend, etc.
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from backend.agent_manager import run_analysis_pipeline

# async def main():
#     """
#     Runs a test of the full agent analysis pipeline with a sample transcript
#     and prints the results from each agent.
#     """
#     print("--- 🧪 Starting Agent Test ---")

#     # This sample transcript is designed to trigger several agents.
#     sample_transcript = (
#         "In my last project, I was completely wrong about our data strategy. "
#         "I had insisted we use a specific database, but my junior colleague "
#         "showed me a better, more scalable option. I had to change my mind, "
#         "and I'm glad I did. I learned a lot from them. We ended up succeeding, "
#         "and I made sure to give them full credit. I don't know what we would "
#         "have done without their insight."
#     )

#     print(f"\n📜 Sample Transcript:\n\"{sample_transcript}\"")
#     print("\n🚀 Running analysis pipeline...")

#     # Run the same pipeline the backend uses
#     try:
#         results = await run_analysis_pipeline(sample_transcript)

#         print("\n--- ✅ Analysis Complete. Results: ---")
#         if not results:
#             print("No results returned from the pipeline.")
#             return

#         for score in sorted(results, key=lambda x: x.agent_name):
#             # Format output for readability
#             status = "✅" if score.evidence != "Agent execution failed." else "❌"
#             print(f"\n{status} Agent: {score.agent_name}")
#             print(f"   Score: {score.score}")
#             print(f"   Evidence: \"{score.evidence}\"")

#     except Exception as e:
#         print(f"\n--- ❌ An error occurred during the agent test: ---")
#         print(e)

#     finally:
#         print("\n--- 🧪 Agent Test Finished ---")

# if __name__ == "__main__":
#     # In case the .env file isn't loaded automatically
#     from dotenv import load_dotenv
#     load_dotenv()
    
#     asyncio.run(main()) 


























# # test_agents.py                            
# # test_agents.py
# import asyncio
# import sys
# import os

# # Add project root to Python path to allow imports from core, backend, etc.
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from backend.agent_manager import run_analysis_pipeline

# async def main():
#     print("--- 🧪 Starting Agent Test ---\n")
    
#     sample = (
#         "In my last project, I was completely wrong about our data strategy. "
#         "I had insisted we use a specific database, but my junior colleague "
#         "showed me a better, more scalable option. I had to change my mind, "
#         "and I'm glad I did. I learned a lot from them. We ended up succeeding, "
#         "and I made sure to give them full credit. I don't know what we would "
#         "have done without their insight."
#     )
    
#     print("📜 Sample Transcript:\n", sample, "\n")
#     print("🚀 Running analysis pipeline...\n")
    
#     try:
#         res = await run_analysis_pipeline(sample)
        
#         print("--- ✅ Analysis Complete. Results: ---\n")
        
#         # Print results in a clean, aligned format
#         print(f"{'Humility Score':<20}: {res.humility_score:.1f}")
#         print(f"{'Humility Evidence':<20}: {res.humility_evidence}\n")
        
#         print(f"{'Learning Score':<20}: {res.learning_score:.1f}")
#         print(f"{'Learning Evidence':<20}: {res.learning_evidence}\n")
        
#         print(f"{'Feedback Score':<20}: {res.feedback_score:.1f}")
#         print(f"{'Feedback Evidence':<20}: {res.feedback_evidence}\n")
        
#         print(f"{'Mistakes Score':<20}: {res.mistakes_score:.1f}")
#         print(f"{'Mistakes Evidence':<20}: {res.mistakes_evidence}\n")
        
#         # Calculate and display overall score
#         overall = (
#             res.humility_score +
#             res.learning_score +
#             res.feedback_score +
#             res.mistakes_score
#         ) / 4.0
        
#         print(f"{'Overall (avg)':<20}: {overall:.1f}/10")
        
#     except Exception as e:
#         print(f"\n❌ Error during analysis: {e}")
#     finally:
#         print("\n--- 🧪 Agent Test Finished ---")

# if __name__ == "__main__":
#     # Load .env if available
#     try:
#         from dotenv import load_dotenv
#         load_dotenv()
#     except ImportError:
#         pass
    
#     asyncio.run(main())






















# # test_agents.py
# import asyncio
# import sys
# import os

# # Add project root to Python path to allow imports from core, backend, etc.
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from backend.agent_manager import run_analysis_pipeline

# async def main():
#     """
#     Runs a test of the full agent analysis pipeline with a sample transcript
#     and prints the results from each agent.
#     """
#     print("--- 🧪 Starting Agent Test ---")

#     # This sample transcript is designed to trigger several agents.
#     sample_transcript = (
#         "In my last project, I was completely wrong about our data strategy. "
#         "I had insisted we use a specific database, but my junior colleague "
#         "showed me a better, more scalable option. I had to change my mind, "
#         "and I'm glad I did. I learned a lot from them. We ended up succeeding, "
#         "and I made sure to give them full credit. I don't know what we would "
#         "have done without their insight."
#     )

#     print(f"\n📜 Sample Transcript:\n\"{sample_transcript}\"")
#     print("\n🚀 Running analysis pipeline...")

#     # Run the same pipeline the backend uses
#     try:
#         results = await run_analysis_pipeline(sample_transcript)

#         print("\n--- ✅ Analysis Complete. Results: ---")
#         if not results:
#             print("No results returned from the pipeline.")
#             return

#         for score in sorted(results, key=lambda x: x.agent_name):
#             # Format output for readability
#             status = "✅" if score.evidence != "Agent execution failed." else "❌"
#             print(f"\n{status} Agent: {score.agent_name}")
#             print(f"   Score: {score.score}")
#             print(f"   Evidence: \"{score.evidence}\"")

#     except Exception as e:
#         print(f"\n--- ❌ An error occurred during the agent test: ---")
#         print(e)

#     finally:
#         print("\n--- 🧪 Agent Test Finished ---")

# if __name__ == "__main__":
#     # In case the .env file isn't loaded automatically
#     from dotenv import load_dotenv
#     load_dotenv()
    
#     asyncio.run(main()) 




















# test_agents.py
import asyncio
import sys
import os

# Ensure we can import backend/*
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.agent_manager import run_analysis_pipeline


async def main():
    print("--- 🧪 Starting Agent Test ---\n")

    sample = (
        "In my last project, I was completely wrong about our data strategy. "
        "I had insisted we use a specific database, but my junior colleague "
        "showed me a better, more scalable option. I had to change my mind, "
        "and I'm glad I did. I learned a lot from them. We ended up succeeding, "
        "and I made sure to give them full credit. I don't know what we would "
        "have done without their insight."
    )

    print("📜 Sample Transcript:\n", sample, "\n")
    print("🚀 Running analysis pipeline...\n")

    try:
        results = await run_analysis_pipeline(sample)

        print("--- ✅ Analysis Complete. Results: ---\n")
        # Sort by agent_name for consistency
        for entry in sorted(results, key=lambda x: x["agent_name"]):
            name = entry["agent_name"]
            score = entry["score"]
            evidence = entry["evidence"]
            status = "✅" if score != 0 else "❌"

            print(f"{status} {name}")
            print(f"   Score   : {score:.1f}")
            print(f"   Evidence: {evidence}\n")

        # Compute overall average score
        avg = (
            sum(entry["score"] for entry in results) / len(results)
            if results
            else 0.0
        )
        print(f"Overall average score: {avg:.1f}/10")

    except Exception as e:
        print(f"\n--- ❌ Error during agent test: {e}")

    finally:
        print("\n--- 🧪 Agent Test Finished ---")


if __name__ == "__main__":
    # load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(main())
