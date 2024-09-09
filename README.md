# 🍕 Gourmet Vision: AI-Powered Food Detection 🍝

Welcome to Gourmet Vision, where artificial intelligence meets culinary expertise! This project harnesses the power of YOLOv8 to create a sophisticated food detection model capable of identifying thirty six food items

## 🌟 Project Highlights

- 🧠 Utilizes state-of-the-art YOLOv8 architecture
- 🍽️ Specialized in detecting Italian cuisine favorites
- 🚀 Easy-to-use training script for customization
- 📊 Comprehensive dataset with train, validation, and test splits

## 🗂️ Project Structure

Our kitchen of code is organized as follows:

- `data.yaml`: The recipe book - configuration file for dataset paths and class names
- `train.py`: The master chef - Python script that trains our AI food critic

## 🍽️ The Menu (Dataset)

Our AI is trained on a smorgasbord of images, carefully curated and divided:

- 🍳 Train set: `.../train/images`
- 🥘 Validation set: `.../valid/images`
- 🍲 Test set: `.../test/images`

On today's menu, we're serving thirty six delicious treats:
1. 🥤 AW cola
2. 🥩🇨🇳 Beijing Beef
3. 🍜 Chow Mein
4. 🍚 Fried Rice
5. 🥔 Hashbrown
6. 🍤🥜 Honey Walnut Shrimp
7. 🐔🌶️ Kung Pao Chicken
8. 🐔🥬 String Bean Chicken Breast
9. 🥬 Super Greens
10. 🍗🍊 The Original Orange Chicken
11. 🍚 White Steamed Rice
12. 🍚🌶️ Black pepper rice bowl
13. 🍔 Burger
14. 🥕🥚 Carrot eggs
15. 🍔🧀 Cheese burger
16. 🍗🧇 Chicken waffle
17. 🍗 Chicken nuggets
18. 🥬 Chinese cabbage
19. 🌭🇨🇳 Chinese sausage
20. 🌽 Crispy corn
21. 🍛 Curry
22. 🍟 French fries
23. 🍗 Fried chicken
24. 🍗 Fried chicken
25. 🥟 Fried dumplings
26. 🍳 Fried eggs
27. 🥭🐔 Mango chicken pocket
28. 🍔🧀 Mozza burger
29. 🌱 Mung bean sprouts
30. 🍗 Nugget
31. 🥔 Perkedel
32. 🍚 Rice
33. 🥤 Sprite
34. 🧀 Tostitos cheese dip sauce
35. 🥔 Triangle hash brown
36. 🥬 Water spinach


## 👨‍🍳 Training the AI Chef

Our model undergoes rigorous training to become the Gordon Ramsay of food detection:

- 🏋️‍♂️ Training regimen: 250 epochs
- 🍱 Batch size: 16 images at a time
- 👁️ Image size: A crisp 640x640 resolution
- 💪 Hardware: GPU-powered (RTX 3060 Laptop) for lightning-fast learning
- 🛑 Early stopping: We know when the dish is perfectly cooked (patience of 20)

## 🚀 Let's Get Cooking!

To train your very own AI food critic:

1. Ensure your kitchen (development environment) is stocked with the necessary ingredients (dependencies): 
```bash
pip install ultralytics
```

2. Fire up the stove (run the training script):
```bash
python train.py
```

3. Sit back and watch as your AI chef learns the art of food detection. The fruits of its labor (trained model and results) will be plated up in `.../food-model`.

## 🍱 Dataset: The Secret Sauce

Our gourmet dataset is sourced from the prestigious Roboflow kitchen:

- 👨‍🍳 Head Chef (Workspace): kaavin-study-drive
- 🍽️ Signature Dish (Project): food-k8grc
- 🥇 Michelin Stars (Version): 1
- 📜 Recipe Sharing Policy (License): MIT
- 🌐 Reservations (URL): [Book a table at our Roboflow restaurant](https://universe.roboflow.com/kaavin-study-drive/food-k8grc/dataset/1)

## 📜 License to Cook

This project is served under the MIT License. Feel free to savor, modify, and share the recipe, but don't forget to credit the original chefs!

## 🍴 Bon Appétit!

We hope you enjoy using Gourmet Vision as much as we enjoyed creating it. May your code be bug-free and your detections accurate!

Remember, in the world of AI and food, pixels are the new flavors, and neural networks are the new taste buds. Happy coding and happy eating! 🎉👨‍🍳🤖
