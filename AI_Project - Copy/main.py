import os
import tkinter as tk
from tkinter import messagebox, scrolledtext, simpledialog, Toplevel, Canvas, Radiobutton, IntVar
from tkinter import ttk
import random
import openai 
from dotenv import load_dotenv
import io
from PIL import Image, ImageTk
import requests
import threading
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from tkhtmlview import HTMLLabel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from environment variables
openai.api_key = os.getenv('OPENAI_SECRET_KEY')

# Dictionary to store user data
users_db = {}

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def generate_new_prompt(keywords, ingredients):
    # Generate a new prompt for the AI based on user feedback keywords and ingredients
    keyword_str = ' '.join(keywords)
    new_prompt = f"Generate a list of recipes based on these ingredients: {ingredients}, with emphasis on: {keyword_str}."
    return new_prompt

def extract_keywords_from_feedback(feedback):
    # Extract keywords from user feedback
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(feedback)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_words)])
    tfidf_scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0])
    keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:5]
    return [keyword for keyword, score in keywords]

class FeedbackDialog(tk.Toplevel):
    def __init__(self, parent, ingredients):
        super().__init__(parent)
        self.ingredients = ingredients
        self.title("Submit Feedback")
        self.geometry("400x300")
        self.transient(parent)
        self.grab_set()

        self.feedback_label = tk.Label(self, text=f"Feedback for ingredients: {self.ingredients}", font=("Arial", 12))
        self.feedback_label.pack(padx=10, pady=(10, 0))

        self.feedback_text = scrolledtext.ScrolledText(self, width=40, height=10)
        self.feedback_text.pack(padx=10, pady=10)

        submit_button = ttk.Button(self, text="Submit", command=self.submit, style="TButton")
        submit_button.pack(pady=10)

        self.result = None

    def submit(self):
        self.result = self.feedback_text.get("1.0", tk.END).strip()
        self.destroy()

class RecipeApp:
    def __init__(self, master):
        self.master = master
        master.title("AI-Powered Recipe Recommendation App")
        master.geometry("800x600")

        self.style = {"font": ("Arial", 14), "bg": "#e0e0e0"}
        self.init_styles()

        self.initialize_ui()

    def init_styles(self):
        # Initialize styles for the application
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 16), background="#1E3A8A", foreground="blue", borderwidth=1, relief="flat")
        style.map("TButton",
                  foreground=[('pressed', 'red'), ('active', 'blue')],
                  background=[('pressed', '!disabled', 'black'), ('active', 'white')],
                  relief=[('pressed', 'groove'), ('!pressed', 'ridge')])
        style.configure("TButton", padding=6, relief="flat", background="#1E3A8A", borderwidth=0)
        style.layout("TButton",
                     [('Button.border', {'sticky': 'nswe', 'children':
                         [('Button.padding', {'sticky': 'nswe', 'children':
                             [('Button.label', {'sticky': 'nswe'})]
                         })]
                     })])

    def initialize_ui(self):
        # Set up the initial UI components
        self.master.configure(bg="#1E3A8A")

        self.login_frame = tk.Frame(self.master, bg="#1E3A8A")
        self.login_frame.pack(pady=50, padx=20, fill=tk.BOTH, expand=True)

        self.username_label = tk.Label(self.login_frame, text="USERNAME", font=("Arial, 16"), bg="#1E3A8A", fg="white")
        self.username_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.username_entry = tk.Entry(self.login_frame, font=("Arial, 16"), width=25)
        self.username_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.password_label = tk.Label(self.login_frame, text="PASSWORD", font=("Arial, 16"), bg="#1E3A8A", fg="white")
        self.password_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.password_entry = tk.Entry(self.login_frame, font=("Arial, 16"), width=25, show='*')
        self.password_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.login_button = ttk.Button(self.login_frame, text="LOGIN", command=self.login, style="TButton")
        self.login_button.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")

        self.register_button = ttk.Button(self.login_frame, text="REGISTER", command=self.register, style="TButton")
        self.register_button.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")

        self.login_frame.grid_columnconfigure(1, weight=1)

        self.diet_label = tk.Label(self.login_frame, text="DIETARY RESTRICTION", font=("Arial, 16"), bg="#1E3A8A", fg="white")
        self.diet_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.diet_entry = tk.Entry(self.login_frame, font=("Arial, 16"), width=25)
        self.diet_entry.grid(row=4, column=1, padx=10, pady=10)

        self.user_favorites = []
        self.user_dislikes = []
        self.current_user = None
        self.user_history = []
        self.last_searched_ingredients = ""  # Track last searched ingredients

    def login(self):
        # Handle user login
        username = self.username_entry.get()
        password = self.password_entry.get()
        if username in users_db and users_db[username]['password'] == password:
            self.current_user = username
            self.user_favorites = users_db[username].get('favorites', [])
            self.user_dislikes = users_db[username].get('dislikes', [])
            self.user_history = users_db[username].get('history', [])
            self.user_diet = users_db[username].get('diet', [])
            self.update_history_listbox()
            messagebox.showinfo("Login", f"Welcome {username}! Login successful!")
            self.open_recommendation_window()
        else:
            messagebox.showerror("Login", "Login failed. Incorrect username or password.")

    def register(self):
        # Handle user registration
        username = self.username_entry.get()
        password = self.password_entry.get()
        diet = self.diet_entry.get()
        if username and password and username not in users_db:
            users_db[username] = {
                'password': password,
                'favorites': [],
                'dislikes': [],
                'history': [],
                'diet': diet.split(","),
                'feedback': {}
            }
            self.current_user = username
            self.user_favorites = []
            self.user_dislikes = []
            self.user_diet = users_db[username]['diet']
            messagebox.showinfo("Register", "Registration successful!")
            self.open_recommendation_window()
        else:
            messagebox.showerror("Register", "Registration failed. User already exists or invalid username.")

    def logout(self):
        # Handle user logout
        if self.current_user:
            users_db[self.current_user]['history'] = self.user_history
        self.current_user = None
        self.user_favorites = []
        self.user_dislikes = []
        self.user_history = []
        self.user_diet = []
        messagebox.showinfo("Logout", "You have been logged out successfully.")
        self.clear_ui()

    def open_recommendation_window(self):
        # Open the recommendation window after login
        self.recommendation_window = Toplevel(self.master)
        self.recommendation_window.title(f"Welcome {self.current_user} - AI-Powered Recipe Recommendations")

        screen_width = self.recommendation_window.winfo_screenwidth()
        screen_height = self.recommendation_window.winfo_screenheight()

        x = (screen_width / 2) - (1200 / 2)
        y = (screen_height / 2) - (800 / 2)
        self.recommendation_window.geometry(f'1200x650+{int(x)}+{int(y)}')

        self.setup_recommendation_ui()
        self.update_history_listbox()

    def setup_recommendation_ui(self):
        # Set up the UI components for the recommendation window
        self.recommendation_window.configure(bg="#e0e0e0")

        main_frame = tk.Frame(self.recommendation_window, bg="#e0e0e0")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.image_canvas = tk.Canvas(main_frame, width=300, height=300, bg='white')
        self.image_canvas.pack(side=tk.LEFT, padx=10, pady=20, fill=tk.BOTH, expand=False)

        self.result_label = HTMLLabel(main_frame, width=40, height=10, font=("Arial", 14), wrap=tk.WORD)
        self.result_label.pack(side=tk.RIGHT, padx=10, pady=20, fill=tk.BOTH, expand=False)

        self.history_listbox = tk.Listbox(main_frame, width=40, height=10, font=("Arial", 14))
        self.history_listbox.pack(side=tk.LEFT, padx=10, pady=20, fill=tk.BOTH, expand=False)
        self.history_listbox.bind('<<ListboxSelect>>', self.display_selected_history_threaded)

        bottom_frame = tk.Frame(self.recommendation_window, bg="#e0e0e0")
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.entry_label = tk.Label(bottom_frame, text="Enter your favorite ingredients", font=("Arial", 16), bg="#e0e0e0")
        self.entry_label.pack(pady=10)

        self.entry = tk.Entry(bottom_frame, font=("Arial", 16), width=60)
        self.entry.pack(pady=10)

        button_row1 = tk.Frame(bottom_frame, bg="#e0e0e0")
        button_row1.pack(pady=10, fill=tk.BOTH, expand=True)
        button_row2 = tk.Frame(bottom_frame, bg="#e0e0e0")
        button_row2.pack(pady=10, fill=tk.BOTH, expand=True)

        self.dislike_button = ttk.Button(button_row1, text="Dislike Recipe", command=self.dislike_recipe, style="TButton")
        self.dislike_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.recommend_button = ttk.Button(button_row1, text="Recommend Recipes", command=self.recommend_recipes, style="TButton")
        self.recommend_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.combined_graph_button = ttk.Button(button_row2, text="Show Combined Graph", command=self.show_combined_graph, style="TButton")
        self.combined_graph_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.add_favorites_button = ttk.Button(button_row1, text="Add to Favorites", command=self.add_current_recipe_to_favorites, style="TButton")
        self.add_favorites_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.logout_button = ttk.Button(button_row2, text="Logout", command=self.logout, style="TButton")
        self.logout_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.feedback_button = ttk.Button(button_row2, text="Submit Feedback", command=self.submit_feedback, style="TButton")
        self.feedback_button.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

    def dislike_recipe(self):
        # Add the currently displayed recipe to the user's disliked recipes
        current_recipe = self.get_current_displayed_recipe()
        if current_recipe:
            if current_recipe not in self.user_dislikes:
                self.user_dislikes.append(current_recipe)
                users_db[self.current_user]['dislikes'] = self.user_dislikes
                messagebox.showinfo("Dislike", "Recipe added to dislikes successfully!")
            else:
                messagebox.showinfo("Dislike", "Recipe is already in dislikes.")
        else:
            messagebox.showerror("Error", "No recipe available to dislike.")

    def add_current_recipe_to_favorites(self):
        # Add the currently displayed recipe to the user's favorite recipes
        current_recipe = self.get_current_displayed_recipe()
        if current_recipe:
            self.add_to_favorites(current_recipe)
        else:
            messagebox.showerror("Error", "No recipe available to add to favorites.")

    def get_current_displayed_recipe(self):
        # Get the currently displayed recipe
        if hasattr(self, 'current_displayed_recipe'):
            return self.current_displayed_recipe
        else:
            return None

    def add_to_favorites(self, recipe):
        # Add a recipe to the user's favorites
        if self.current_user and recipe:
            if recipe not in users_db[self.current_user]['favorites']:
                users_db[self.current_user]['favorites'].append(recipe)
                messagebox.showinfo("Favorites", "Recipe added to favorites successfully!")
            else:
                messagebox.showinfo("Favorites", "Recipe is already in favorites.")
        else:
            messagebox.showerror("Favorites", "No user logged in or no recipe specified.")

    def display_selected_history_threaded(self, event):
        # Display selected history in a new thread
        threading.Thread(target=self.display_selected_history, args=(event,)).start()

    def display_selected_history(self, event):
        # Display the selected history item
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            data = self.user_history[index]
            ingredients = ', '.join(data['ingredients']) if isinstance(data['ingredients'], list) else data['ingredients']
            recipes = data['recipes']
            formatted_recipes = ""
            for i, recipe in enumerate(recipes, start=1):
                if ":" in recipe:
                    title, details = recipe.split(":", 1)
                    formatted_recipes += f"<li><strong style='font-size:16px;'>{title.strip()}:</strong> {details.strip()}</li>"
                else:
                    formatted_recipes += f"<li>{recipe.strip()}</li>"
            display_text = f"""
            <p><strong style="font-size:18px;">Ingredients:</strong> {ingredients}</p>
            <p><strong style="font-size:18px;">Recipes:</strong></p>
            <ul>{formatted_recipes}</ul>
            """
            self.result_label.set_html(display_text)

            self.image_canvas.delete("all")
            self.image_canvas.create_text(150, 150, text="Loading image...", fill="black", font=("Arial", 16))

            self.thread_safe_update_image(data['image_data'])

    def thread_safe_update_image(self, image_data):
        # Update the image in a thread-safe manner
        self.master.after(0, self.update_image, image_data)

    def update_image(self, image_data):
        # Update the image on the canvas
        if image_data:
            try:
                if image_data.startswith('images/'):  # Check if the image data is a local path
                    image_path = os.path.abspath(image_data)
                    image = Image.open(image_path)
                    image = image.resize((300, 300))  # Resize the image if needed
                    img_tk = ImageTk.PhotoImage(image)
                    self.image_canvas.delete("all")
                    self.image_canvas.create_image(150, 150, image=img_tk)
                    self.image_canvas.image = img_tk
                else:
                    # Save the generated image to the local folder
                    random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    image_filename = f"images/{self.current_user}_{self.last_searched_ingredients}_{random_chars}.png"
                    image.save(image_filename, "PNG")
                    self.thread_safe_update_image(image_filename)  # Update the image in the UI
            except Exception as e:
                self.image_canvas.delete("all")
                self.image_canvas.create_text(150, 150, text="Failed to load image", fill="red", font=("Arial", 16))
        else:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(150, 150, text="No image available.", fill="red", font=("Arial", 16))

    def update_history_listbox(self):
        # Update the history listbox with user history
        if hasattr(self, 'history_listbox') and self.history_listbox.winfo_exists():
            self.history_listbox.delete(0, tk.END)
            for history in self.user_history:
                feedback_flag = " + feedback" if history.get('from_feedback') else ""
                self.history_listbox.insert(tk.END, f"{history['ingredients']} - {len(history['recipes'])} recipes{feedback_flag}")

    def recommend_recipes(self):
        # Recommend recipes based on user input
        self.recommend_button.config(text="Loading...", state=tk.DISABLED)

        def fetch_data():
            ingredients = self.entry.get()
            if ingredients:
                self.last_searched_ingredients = ingredients  # Track last searched ingredients
                diet_restrictions = ', '.join(self.user_diet)
                prompt = f"Generate a list of recipes based on these ingredients: {ingredients}, considering likes: {self.user_favorites}, dislikes: {self.user_dislikes}, and dietary restrictions: {diet_restrictions}."
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "Recipe recommendation system"},
                            {"role": "user", "content": prompt}]
                )
                recipes = response['choices'][0]['message']['content'].strip().split('\n')
                formatted_recipes = ""
                for i, recipe in enumerate(recipes, start=1):
                    if ":" in recipe:
                        title, details = recipe.split(":", 1)
                        formatted_recipes += f"<li><strong style='font-size:16px;'>{title.strip()}:</strong> {details.strip()}</li>"
                    else:
                        formatted_recipes += f"<li>{recipe.strip()}</li>"
                display_text = f"""
                <p><strong style="font-size:18px;">AI Recommended Recipes for {ingredients}:</strong></p>
                <ul>{formatted_recipes}</ul>
                """
                if not recipes:
                    display_text = "<p>No matching recipes found.</p>"

                self.recommendation_window.after(0, self.update_ui, display_text, ingredients, recipes)
                if recipes:
                    self.current_displayed_recipe = recipes[0]

            else:
                messagebox.showinfo("Input Required", "Please enter some ingredients to get AI-based recommendations.")

            self.recommendation_window.after(0, self.show_dislike_button)
            self.recommendation_window.after(0, self.restore_buttons)

        thread = threading.Thread(target=fetch_data)
        thread.start()

    def update_ui(self, display_text, ingredients, recipes):
        # Update the UI with the recommended recipes
        self.result_label.set_html(display_text)
        image_data = self.generate_image_data(ingredients)
        self.user_history.append({'ingredients': ingredients, 'recipes': recipes, 'diet': self.user_diet, 'image_data': image_data, 'from_feedback': False})
        self.update_history_listbox()
        users_db[self.current_user]['history'] = self.user_history
        self.thread_safe_update_image(image_data)

    def show_dislike_button(self):
        # Show the dislike button
        self.dislike_button.pack(pady=10)

    def restore_buttons(self):
        # Restore the state of the buttons
        self.recommend_button.config(text="Recommend Recipes", state=tk.NORMAL)
        self.dislike_button.config(text="Dislike Recipe", state=tk.NORMAL)

    def generate_image_data(self, description):
        # Generate image data using OpenAI's image generation API
        try:
            response = openai.Image.create(
                prompt=f"A photo of {description}",
                n=1,
                size="1024x1024"
            )
            image_data = response['data'][0]['url']
            random_chars = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            image_filename = f"images/{self.current_user}_{description}_{random_chars}.png"            
            with open(image_filename, 'wb') as file:
                image_response = requests.get(image_data)
                file.write(image_response.content)
                return image_filename
        except Exception as e:
            return None

    def clear_ui(self):
        # Clear the UI components
        for widget in self.master.winfo_children():
            widget.destroy()

        self.initialize_ui()

    def submit_feedback(self):
        # Submit feedback for the recommended recipes
        if self.last_searched_ingredients:
            selection = self.history_listbox.curselection()
            if selection:
                index = selection[0]
                selected_data = self.user_history[index]
                dialog = FeedbackDialog(self.recommendation_window, selected_data['ingredients'])
                dialog.update_idletasks()
                x = (self.master.winfo_screenwidth() - dialog.winfo_reqwidth()) // 2
                y = (self.master.winfo_screenheight() - dialog.winfo_reqheight()) // 2
                dialog.geometry("+%d+%d" % (x, y))
                self.recommendation_window.wait_window(dialog)
                feedback = dialog.result
                if feedback:
                    users_db[self.current_user]['feedback'][selected_data['ingredients']] = feedback
                    messagebox.showinfo("Feedback", "Thank you for your feedback!")
                    self.process_feedback(feedback, selected_data['ingredients'])
            else:
                messagebox.showerror("Feedback Error", "No item selected for feedback.")
        else:
            messagebox.showerror("Feedback Error", "No ingredients selected for feedback.")
    
    def process_feedback(self, feedback, ingredients):
        # Process the feedback and generate new recommendations
        keywords = extract_keywords_from_feedback(feedback)
        new_prompt = generate_new_prompt(keywords, ingredients)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Recipe feedback system"},
                    {"role": "user", "content": new_prompt}]
        )
        new_recipes = response['choices'][0]['message']['content'].strip().split('\n')
        formatted_recipes = ""
        for i, recipe in enumerate(new_recipes, start=1):
            if ":" in recipe:
                title, details = recipe.split(":", 1)
                formatted_recipes += f"<li><strong style='font-size:16px;'>{title.strip()}:</strong> {details.strip()}</li>"
            else:
                formatted_recipes += f"<li>{recipe.strip()}</li>"
        display_text = f"""
        <p><strong style="font-size:18px;">New Recommendations based on feedback for {ingredients}:</strong></p>
        <ul>{formatted_recipes}</ul>
        """
        self.result_label.set_html(display_text)
        
        image_data = self.generate_image_data(ingredients)  # Generate image data based on ingredients
        self.thread_safe_update_image(image_data)  # Update the image in the UI
        
        self.user_history.append({'ingredients': ingredients, 'recipes': new_recipes, 'diet': self.user_diet, 'image_data': image_data, 'from_feedback': True})
        self.update_history_listbox()
        users_db[self.current_user]['history'] = self.user_history

    def recommend_recipes_collaborative_filtering(self, ingredients):
        # Recommend recipes using collaborative filtering
        similarities = {}
        for user, data in users_db.items():
            if user != self.current_user:
                common_likes = len(set(data['favorites']) & set(self.user_favorites))
                common_dislikes = len(set(data['dislikes']) & set(self.user_dislikes))
                similarities[user] = common_likes + common_dislikes
        
        similar_users = sorted(similarities, key=similarities.get, reverse=True)[:3]
        
        recommended_recipes = []
        for user in similar_users:
            recommended_recipes.extend(users_db[user]['favorites'])
        
        unique_recommendations = list(set(recommended_recipes) - set(self.user_favorites) - set(self.user_dislikes))
        
        if unique_recommendations:
            formatted_recipes = ""
            for i, recipe in enumerate(unique_recommendations, start=1):
                formatted_recipes += f"<li>{recipe.strip()}</li>"
            display_text = f"""
            <p><strong style="font-size:18px;">Recommended Recipes based on similar users for {ingredients}:</strong></p>
            <ul>{formatted_recipes}</ul>
            """
            self.result_label.set_html(display_text)
            image_data = self.generate_image_data(ingredients)  # Generate image data based on ingredients
            self.thread_safe_update_image(image_data)  # Update the image in the UI
            self.user_history.append({'ingredients': ingredients, 'recipes': unique_recommendations, 'diet': self.user_diet, 'image_data': image_data, 'from_feedback': False})
            self.update_history_listbox()
            users_db[self.current_user]['history'] = self.user_history

    def show_combined_graph(self):
        # Show a graph comparing favorite and disliked recipes
        if self.current_user:
            favorite_recipes = users_db[self.current_user].get('favorites', [])
            disliked_recipes = users_db[self.current_user].get('dislikes', [])

            favorite_counts = Counter(favorite_recipes)
            dislike_counts = Counter(disliked_recipes)

            recipes = list(set(favorite_recipes + disliked_recipes))
            fav_counts = [favorite_counts.get(recipe, 0) for recipe in recipes]
            dis_counts = [dislike_counts.get(recipe, 0) for recipe in recipes]

            fig, ax = plt.subplots(figsize=(10, 5))
            bar_width = 0.35
            index = range(len(recipes))

            bar1 = ax.bar(index, fav_counts, bar_width, label='Favorites', color='b')
            bar2 = ax.bar([p + bar_width for p in index], dis_counts, bar_width, label='Dislikes', color='r')

            ax.set_xlabel('Recipes')
            ax.set_ylabel('Counts')
            ax.set_title('Comparison of Favorite and Disliked Recipes')
            ax.set_xticks([p + bar_width / 2 for p in index])
            ax.set_xticklabels([recipe[:10] for recipe in recipes], rotation=45, ha='right')
            ax.legend()

            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("User Not Logged In", "Please log in to view the recipes graph.")

def main():
    root = tk.Tk()
    app = RecipeApp(root)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x = (screen_width / 2) - (800 / 2)
    y = (screen_height / 2) - (600 / 2)
    root.geometry(f'800x400+{int(x)}+{int(y)}')

    root.mainloop()

if __name__ == "__main__":
    main()
