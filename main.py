def main():
    print("2DGEN default pipeline: token-based diffusion.")
    print("Train:  uv run python 2DGEN/train_tokens.py --csv data/C2DB/c2db_summary.csv")
    print("Sample: uv run python 2DGEN/sample_tokens.py --checkpoint outputs/checkpoints/atomdenoiser_best.pt")


if __name__ == "__main__":
    main()
