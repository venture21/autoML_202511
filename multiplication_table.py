# multiplication_table.py

def print_multiplication_table():
    """
    Prints the multiplication table (구구단) from 1 to 9.
    """
    for i in range(1, 10):
        print(f"--- {i}단 ---")
        for j in range(1, 10):
            print(f"{i} x {j} = {i * j}")
        print() # Add a blank line for better readability between tables

if __name__ == "__main__":
    print_multiplication_table()
