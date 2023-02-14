
print("\033[31mThis is red font.\033[0m")
print("\033[32mThis is green font.\033[0m")
print("\033[33mThis is yellow font.\033[0m")
print("\033[34mThis is blue font.\033[0m")
# More than 37 will display the default font.
print("\033[38mThis is the default font. \033[0m")


class CColor:
    Red = '\033[91m'
    Green = '\u001b[32m'
    Yellow = '\u001b[33m'
    Blue = '\u001b[34m'
    Cyan = '\u001b[36m'
    White = '\033[0m'


Color = CColor

print(f"{Color.Yellow}[*] Hellow World! [*]{Color.White}")
print("Yesss this is white.")
