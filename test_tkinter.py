import tkinter as tk

def main():
    # Create a basic Tkinter window
    root = tk.Tk()
    root.title("Tkinter Test")
    root.geometry("300x200")
    
    # Add a label
    label = tk.Label(root, text="If you can see this, Tkinter is working!")
    label.pack(pady=20)
    
    # Add a button
    button = tk.Button(root, text="Close", command=root.destroy)
    button.pack(pady=20)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    print("Starting Tkinter test...")
    main()
    print("Tkinter test completed")  # This will only print after closing the window 