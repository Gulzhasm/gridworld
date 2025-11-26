#!/usr/bin/env python3
"""
Interactive Frozen Lake Game
Run this script to play the Frozen Lake game interactively.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frozen_lake import FrozenLake, play


def main():
    print("\n" + "="*60)
    print("FROZEN LAKE - INTERACTIVE GAME")
    print("="*60)
    print("\nChoose a difficulty level:\n")
    print("1. Small Lake (4×4) - Easy")
    print("2. Big Lake (8×8) - Hard")
    print("3. Custom Lake")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == '1':
        # Small lake
        lake = [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        print("\n🏔️  Small Lake (4×4) selected!")
        
    elif choice == '2':
        # Big lake
        lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]
        print("\n🏔️  Big Lake (8×8) selected!")
        
    elif choice == '3':
        # Custom lake
        print("\nEnter your custom lake (use & for start, . for frozen, # for hole, $ for goal)")
        print("Example: & . . . / . # . # / . . . # / # . . $")
        print("(Use / to separate rows)")
        
        try:
            lake_input = input("Enter lake: ").strip()
            rows = lake_input.split('/')
            lake = []
            for row in rows:
                tiles = [tile.strip() for tile in row.split()]
                lake.append(tiles)
            print("\n🏔️  Custom Lake selected!")
        except Exception as e:
            print(f" Error parsing lake: {e}")
            return
    
    elif choice == '4':
        print("\nGoodbye! ")
        return
    
    else:
        print("Invalid choice!")
        return
    
    # Ask for difficulty settings
    print("\nDifficulty Settings:")
    print("1. Normal (10% slip)")
    print("2. Easy (0% slip)")
    print("3. Hard (30% slip)")
    
    difficulty = input("Enter difficulty (1-3, default=1): ").strip() or '1'
    
    slip_prob = {
        '1': 0.1,
        '2': 0.0,
        '3': 0.3
    }.get(difficulty, 0.1)
    
    print(f"\n🎮 Starting game with {slip_prob*100:.0f}% slip probability...")
    print("="*60 + "\n")
    
    # Create environment and play
    try:
        env = FrozenLake(lake, slip=slip_prob, max_steps=len(lake)*len(lake[0]), seed=0)
        play(env)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
