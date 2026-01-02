def main():
    print("2dgen default pipeline: token-based diffusion.")
    print("Train:  uv run python -m twodgen.scrip.train_tokens --csv data/C2DB/c2db_summary.csv")
    print("Sample: uv run python -m twodgen.scrip.sample_tokens --checkpoint outputs/checkpoints/atomdenoiser_best.pt")


if __name__ == "__main__":
    main()
