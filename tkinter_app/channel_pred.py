from tkinter import *

import numpy as np

import json
import pickle



root = Tk()
# root.geometry("700x500")
root.title("Channel Prediction")


title_frame = Frame(root)
title_frame.grid(row=0, column=0, padx=10, pady=10)

title = Label(title_frame, text="Channel Prediction", font=("Times", 14))
title.pack()


main_frame = Frame(root)
main_frame.grid(row=1, column=0)


fresh_Frame = LabelFrame(main_frame, text="Fresh")
fresh_Frame.grid(row=0, column=0, padx=10, pady=10)

fresh_label = Label(fresh_Frame, text="Fresh")
fresh_label.grid(row=0, column=0, padx=10, pady=0)

fresh_textbox = Entry(fresh_Frame)
fresh_textbox.grid(row=1, column=0, padx=10, pady=5)



milk_Frame = LabelFrame(main_frame, text="Milk")
milk_Frame.grid(row=0, column=1, padx=10, pady=10)

milk_label = Label(milk_Frame, text="Milk")
milk_label.grid(row=0, column=0, padx=10, pady=0)

milk_textbox = Entry(milk_Frame)
milk_textbox.grid(row=1, column=0, padx=10, pady=5)



grocery_Frame = LabelFrame(main_frame, text="Grocery")
grocery_Frame.grid(row=0, column=2, padx=10, pady=10)

grocery_label = Label(grocery_Frame, text="Grocery")
grocery_label.grid(row=0, column=0, padx=10, pady=0)

grocery_textbox = Entry(grocery_Frame)
grocery_textbox.grid(row=1, column=0, padx=10, pady=5)



frozen_Frame = LabelFrame(main_frame, text="Frozen")
frozen_Frame.grid(row=0, column=3, padx=10, pady=10)

frozen_label = Label(frozen_Frame, text="Frozen")
frozen_label.grid(row=0, column=0, padx=10, pady=0)

frozen_textbox = Entry(frozen_Frame)
frozen_textbox.grid(row=1, column=0, padx=10, pady=5)



detergent_Frame = LabelFrame(main_frame, text="Detergent paper")
detergent_Frame.grid(row=0, column=4, padx=10, pady=10)

detergent_label = Label(detergent_Frame, text="Detergent paper")
detergent_label.grid(row=0, column=0, padx=10, pady=0)

detergent_textbox = Entry(detergent_Frame)
detergent_textbox.grid(row=1, column=0, padx=10, pady=5)



delicassen_Frame = LabelFrame(main_frame, text="Delicassen")
delicassen_Frame.grid(row=0, column=5, padx=10, pady=10)

delicassen_label = Label(delicassen_Frame, text="Delicassen")
delicassen_label.grid(row=0, column=0, padx=10, pady=0)

delicassen_textbox = Entry(delicassen_Frame)
delicassen_textbox.grid(row=1, column=0, padx=10, pady=5)


# Display label
fresh_display_label = Label(fresh_Frame, text="", font=("Times", 11))
milk_display_label = Label(milk_Frame, text="", font=("Times", 11))
grocery_display_label = Label(grocery_Frame, text="", font=("Times", 11))
frozen_display_label = Label(frozen_Frame, text="", font=("Times", 11))
detergent_display_label = Label(detergent_Frame, text="", font=("Times", 11))
delcasen_display_label = Label(delicassen_Frame, text="", font=("Times", 11))

fresh_display_label.grid(row=2, column=0, padx=10, pady=0)
milk_display_label.grid(row=2, column=0, padx=10, pady=0)
grocery_display_label.grid(row=2, column=0, padx=10, pady=0)
frozen_display_label.grid(row=2, column=0, padx=10, pady=0)
detergent_display_label.grid(row=2, column=0, padx=10, pady=0)
delcasen_display_label.grid(row=2, column=0, padx=10, pady=0)


global _array_columns
global svc_model


with open("model/array_columns.json", "r") as file:
    _array_columns = json.load(file)["array_columns"]

with open("model/client_data_model.pickle", "rb") as file:
    svc_model = pickle.load(file)

#print("model loading complete...")


def predict():
    result: list = []
    x = np.zeros(len(_array_columns))

    x[0] = fresh_textbox.get()
    x[1] = milk_textbox.get()
    x[2] = grocery_textbox.get()
    x[3] = frozen_textbox.get()
    x[4] = detergent_textbox.get()
    x[5] = delicassen_textbox.get()

    fresh_display_label.configure(text=x[0])
    milk_display_label.configure(text=x[1])
    grocery_display_label.configure(text=x[2])
    frozen_display_label.configure(text=x[3])
    detergent_display_label.configure(text=x[4])
    delcasen_display_label.configure(text=x[5])

    #prediction = print(_model.predict([x]))
    result.append(svc_model.predict([x]))
    
    predict_label.configure(text=result[0][0])

    if result[0][0] == 1:
        predict_desc.configure(text="client purchase through Hotel, Restaurant or cafe")
    elif result[0][0] == 2:
        predict_desc.configure(text="Client purchase through Retail Stores")
        

    fresh_textbox.delete(0, END)
    milk_textbox.delete(0, END)
    grocery_textbox.delete(0, END)
    frozen_textbox.delete(0, END)
    detergent_textbox.delete(0, END)
    delicassen_textbox.delete(0, END)



# button

pred_button = Button(
    root,
    text="Predict",
    font=("Times", 12, "bold"),
    command=predict,
    bg="gold4",
    fg="aquamarine"
    
)
pred_button.grid(row=2, column=0, padx=10, pady=10, ipadx=20, ipady=5)


predict_label = Label(
    root,
    font=("Heveltica", 14, "bold"),
    text=""
)
predict_label.grid(row=3, column=0, padx=10, pady=10, ipadx=20, ipady=5)

predict_desc = Label(
    root,
    font=("Heveltica", 14, "bold"),
    text=""
)
predict_desc.grid(row=4, column=0, padx=10, pady=10, ipadx=20, ipady=5)







root.mainloop()