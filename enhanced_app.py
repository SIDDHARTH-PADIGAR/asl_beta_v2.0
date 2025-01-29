import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from enum import Enum
import random
import pyttsx3
import threading


class AppMode(Enum):
    ALPHABET_PRACTICE = 1
    WORD_BUILDER = 2
    QUIZ_MODE = 3
    STORY_TIME = 4

class SignLanguageApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        # Load the model
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']

        self.practice_words = ["HELLO", "THANK", "YOU", "PLEASE", "GOOD", "BAD", "YES", "NO"]
        self.stories = [
            {"text": "I AM HAPPY", "prompt": "Express your feelings"},
            {"text": "HOW ARE YOU", "prompt": "Ask about someone's wellbeing"},
            {"text": "NICE TO MEET", "prompt": "Greet someone new"}
        ]

        # New variables for word builder mode
        self.current_target_word = None
        self.word_builder_progress = 0

        # New variables for story mode
        self.current_story = None
        self.story_progress = 0
        self.current_story_prompt = None

        #text-to-speech engine initalization
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        self.engine.setProperty('volume', 0.9)
        self.speech_thread = None
        self.speech_lock = threading.Lock()

        self.last_letter_time = time.time()

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Application state
        self.current_mode = AppMode.ALPHABET_PRACTICE
        self.scores = {
            'alphabet': {'correct': 0, 'attempts': 0},
            'word': {'correct': 0, 'attempts': 0, 'words_completed': 0},
            'quiz': {'correct': 0, 'attempts': 0, 'streak': 0},
            'story': {'correct': 0, 'attempts': 0, 'stories_completed': 0}
        }
        self.last_prediction_time = time.time()
        self.prediction_buffer = 2.0
        self.attempts = 0
        self.current_word = ""
        self.target_letter = None
        self.collected_letters = []
        self.feedback_timer = 0
        self.show_feedback = False

        self.feedback_text = "Great job! Keep going!"  # For correct answers
        self.feedback_text = f"Almost there! Try making the letter {self.target_letter} again"  # For incorrect

        # Words for practice
        self.practice_words = ["HELLO", "THANK", "YOU", "PLEASE", "GOOD", "BAD", "YES", "NO"]
        self.stories = ["I AM HAPPY", "HOW ARE YOU", "NICE TO MEET"]

        # Letter mapping
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
            22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }

    def speak_text(self, text):
        """Speak text in a separate thread to avoid blocking the main application"""
        def speak_worker():
            with self.speech_lock:
                self.engine.say(text)
                self.engine.runAndWait()

        # Cancel and ongoing speech to avoid speech overlapping
        if self.speech_thread and self.speech_thread.is_alive():
            self.engine.stop()
            self.speech_thread.join()

        #Start new speech thread
        self.speech_thread = threading.Thread(target=speak_worker)
        self.speech_thread.start()



    def draw_ui_panel(self, frame):
        frame_height, frame_width = frame.shape[:2]

        # Top panel with more height
        panel_height = 120
        cv2.rectangle(frame, (0, 0), (frame_width, panel_height),
                      (255, 255, 255), -1)

        # Wider spacing between buttons
        button_width = (frame_width - 200) // 4  # Reduce button width
        for i, mode in enumerate(AppMode):
            x1 = 50 + i * (button_width + 40)  # Add 40px spacing between buttons
            x2 = x1 + button_width
            y1 = 30  # Move buttons down
            y2 = panel_height - 30  # Leave space at bottom

            if mode == self.current_mode:
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (120, 220, 120), -1)
    def draw_feedback_panel(self, frame):
        if self.current_mode == AppMode.ALPHABET_PRACTICE:
            # Draw target letter panel
            cv2.rectangle(frame, (20, 70), (200, 150), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 70), (200, 150), (200, 200, 200), 2)
            if self.target_letter:
                cv2.putText(frame, "Target Letter:", (30, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                cv2.putText(frame, self.target_letter, (80, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        elif self.current_mode == AppMode.WORD_BUILDER:
            # Draw word building panel
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (200, 200, 200), 2)
            cv2.putText(frame, "Current Word:", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(frame, ''.join(self.collected_letters), (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        elif self.current_mode == AppMode.QUIZ_MODE:
            # Draw quiz panel
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (200, 200, 200), 2)
            cv2.putText(frame, f"Score: {self.scores['quiz']['correct']}/{self.scores['quiz']['attempts']}", (30, 95),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            if self.target_letter:
                cv2.putText(frame, f"Show letter: {self.target_letter}", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        elif self.current_mode == AppMode.STORY_TIME:
            # Draw story panel
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (255, 255, 255), -1)
            cv2.rectangle(frame, (20, 70), (frame.shape[1] - 20, 150), (200, 200, 200), 2)
            cv2.putText(frame, "Story:", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            if self.current_word:
                cv2.putText(frame, self.current_word, (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    def process_hand_landmarks(self, frame):
        data_aux = []
        x_ = []
        y_ = []
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect x, y coordinates first
                for i in range(21):  # Use only first 21 landmarks
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Create exactly 42 features (21 landmarks × 2 coordinates)
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                prediction = self.model.predict([np.asarray(data_aux)])
                predicted_character = self.labels_dict[int(prediction[0])]

                # Rest of the drawing code remains the same...

            # Draw bounding box with predicted character
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw a more aesthetic bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
            cv2.rectangle(frame, (x1, y1 - 40), (x1 + 40, y1), (120, 220, 120), -1)
            cv2.putText(frame, predicted_character, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            return frame, predicted_character
        return frame, None

    def handle_alphabet_practice(self, predicted_char):
        if self.target_letter is None:
            self.target_letter = random.choice(list(self.labels_dict.values()))
            self.speak_text(f"Show me the letter {self.target_letter}")


        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            # Only increment attempts for new predictions
            if not hasattr(self, 'last_prediction') or predicted_char != self.last_prediction:
                self.scores['alphabet']['attempts'] += 1
                if predicted_char == self.target_letter:
                    self.scores['alphabet']['correct'] += 1
                    self.target_letter = random.choice(list(self.labels_dict.values()))
                    self.feedback_text = "Great job! Keep going!"
                    self.speak_text("Correct! Great job!")
                else:
                    self.feedback_text = f"Try again! Make the letter: {self.target_letter}"
                    self.speak_text("Try again")

                self.last_prediction = predicted_char
            self.last_prediction_time = current_time

    def handle_word_builder(self, predicted_char):

        # Initialize target word if none exists
        if self.current_target_word is None:
            self.current_target_word = random.choice(self.practice_words)
            self.collected_letters = []
            self.word_builder_progress = 0
            self.speak_text(f"Spell the word: {' '.join(self.current_target_word)}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            try:
                # Check if the predicted character matches the next expected letter
                expected_letter = self.current_target_word[self.word_builder_progress]

                if predicted_char == expected_letter:
                    self.collected_letters.append(predicted_char)
                    self.word_builder_progress += 1
                    self.scores['word']['correct'] += 1
                    self.feedback_text = "Correct! Keep going!"

                    # Check if word is complete
                    if self.word_builder_progress == len(self.current_target_word):
                        self.scores['word']['words_completed'] += 1
                        self.feedback_text = f"Congratulations! You completed the word: {self.current_target_word}"
                        self.speak_text("Word completed! Great job!")
                        self.current_target_word = None # Reset for next word
                        self.current_target_word = random.choice(self.practice_words)
                        self.collected_letters = []
                        self.word_builder_progress = 0
                        self.speak_text(f"Next word: {' '.join(self.current_target_word)}")
                else:
                    self.feedback_text = f"Try again! Next letter should be: {expected_letter}"
                    self.speak_text("Incorrect letter, try again")

                self.scores['word']['attempts'] += 1
                self.last_prediction_time = current_time

            except Exception as e:
                print(f"Word builder error: {e}")
                self.collected_letters = []

    def handle_quiz_mode(self, predicted_char):
        # Initialize target letter if none exists
        if self.target_letter is None:
            self.target_letter = random.choice(list(self.labels_dict.values()))
            self.speak_text(f"Show me the letter {self.target_letter}")

        current_time = time.time()
        try:
            if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
                self.scores['quiz']['attempts'] += 1
                if predicted_char == self.target_letter:
                    self.scores['quiz']['correct'] += 1
                    self.scores['quiz']['streak'] += 1
                    self.feedback_text = f"Correct! Streak: {self.scores['quiz']['streak']}"
                    self.speak_text("Correct! Great job!")
                    # Generate new target letter
                    new_letter = random.choice(list(self.labels_dict.values()))
                    while new_letter == self.target_letter:  # Avoid repeating the same letter
                        new_letter = random.choice(list(self.labels_dict.values()))
                    self.target_letter = new_letter
                    self.speak_text(f"Next letter is {self.target_letter}")
                else:
                    self.scores['quiz']['streak'] = 0
                    self.feedback_text = f"Try again! Show the letter: {self.target_letter}"
                    self.speak_text("Try again")
                self.last_prediction_time = current_time
        except Exception as e:
            print(f"Quiz mode error: {e}")
            self.target_letter = random.choice(list(self.labels_dict.values()))

    def handle_story_mode(self, predicted_char):
        if self.current_story is None:
            self.current_story = random.choice(self.stories)
            self.story_progress = 0
            self.collected_letters = []
            self.current_story_prompt = self.current_story["prompt"]
            self.speak_text(f"{self.current_story_prompt}. Sign: {self.current_story['text']}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            try:
                # Get the current word from the story
                story_words = self.current_story["text"].split()
                current_word = story_words[self.story_progress]

                # Check if we need to start a new word
                if len(self.collected_letters) == len(current_word):
                    self.story_progress += 1
                    self.collected_letters = []
                    if self.story_progress >= len(story_words):
                        # Story completed
                        self.scores['story']['stories_completed'] += 1
                        self.feedback_text = "Story completed! Great job!"
                        self.speak_text("Story completed! Well done!")
                        # Reset for next story
                        self.current_story = random.choice(self.stories)
                        self.story_progress = 0
                        self.collected_letters = []
                        self.current_story_prompt = self.current_story["prompt"]
                        self.speak_text(f"New story! {self.current_story_prompt}")
                        return

                # Check if the predicted character matches the next expected letter
                expected_letter = current_word[len(self.collected_letters)]
                if predicted_char == expected_letter:
                    self.collected_letters.append(predicted_char)
                    self.scores['story']['correct'] += 1
                    self.feedback_text = "Correct! Keep going!"
                else:
                    self.feedback_text = f"Try again! Expected letter: {expected_letter}"
                    self.speak_text("Incorrect letter, try again")

                self.scores['story']['attempts'] += 1
                self.last_prediction_time = current_time

            except Exception as e:
                print(f"Story mode error: {e}")
                self.current_story = random.choice(self.stories)
                self.story_progress = 0
                self.collected_letters = []

    def draw_mode_info(self, frame):
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Draw semi-transparent overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw mode name
        cv2.putText(frame, f"Mode: {self.current_mode.name}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw score based on mode
        if self.current_mode == AppMode.ALPHABET_PRACTICE:
            score_text = f"Score: {self.scores['alphabet']['correct']}/{self.scores['alphabet']['attempts']}"
            cv2.putText(frame, score_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if self.target_letter:
                cv2.putText(frame, f"Show Letter: {self.target_letter}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        elif self.current_mode == AppMode.WORD_BUILDER:
            if self.current_target_word:
                target_text = f"Target Word: {self.current_target_word}"
                current_text = f"Your Progress: {''.join(self.collected_letters)}"
                score_text = f"Words: {self.scores['word']['words_completed']} Correct: {self.scores['word']['correct']}/{self.scores['word']['attempts']}"

                cv2.putText(frame, target_text, (20,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
                cv2.putText(frame, current_text, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, score_text, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        elif self.current_mode == AppMode.QUIZ_MODE:
            if self.target_letter:
                target_text = f"Show Letter: {self.target_letter}"
                score_text = f"Score: {self.scores['quiz']['correct']}/{self.scores['quiz']['attempts']}"
                streak_text = f"Current Streak: {self.scores['quiz']['streak']}"

                cv2.putText(frame, target_text, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, score_text, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, streak_text, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Draw feedback if available
        if hasattr(self, 'feedback_text') and self.feedback_text:
            cv2.putText(frame, self.feedback_text,
                        (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 100, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, predicted_char = self.process_hand_landmarks(frame)

            # Handle mode logic
            try:
                if self.current_mode == AppMode.ALPHABET_PRACTICE:
                    self.handle_alphabet_practice(predicted_char)
                elif self.current_mode == AppMode.WORD_BUILDER:
                    self.handle_word_builder(predicted_char)
                elif self.current_mode == AppMode.QUIZ_MODE:
                    self.handle_quiz_mode(predicted_char)
                elif self.current_mode == AppMode.STORY_TIME:
                    self.handle_story_mode(predicted_char)  # Add story mode handling

                # Draw UI
                self.draw_mode_info(frame)

            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

            cv2.imshow('Sign Language Learning System', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset scores
                self.scores = {
                    'alphabet': {'correct': 0, 'attempts': 0},
                    'word': {'correct': 0, 'attempts': 0, 'words_completed': 0},
                    'quiz': {'correct': 0, 'attempts': 0, 'streak': 0},
                    'story': {'correct': 0, 'attempts': 0, 'stories_completed': 0}
                }
            # Add mode switching
            elif key == ord('1'):
                self.current_mode = AppMode.ALPHABET_PRACTICE
                self.speak_text("Switching to Alphabet Practice mode")
            elif key == ord('2'):
                self.current_mode = AppMode.WORD_BUILDER
                self.speak_text("Switching to Word Builder mode")
            elif key == ord('3'):
                self.current_mode = AppMode.QUIZ_MODE
                self.speak_text("Switching to Quiz mode")
            elif key == ord('4'):
                self.current_mode = AppMode.STORY_TIME
                self.speak_text("Switching to Story Time mode")

        # Cleanup code
        if self.speech_thread and self.speech_thread.is_alive():
            self.engine.stop()
            self.speech_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()