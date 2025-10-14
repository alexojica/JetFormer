import torch


@torch.no_grad()
def _build_model(d_model=64, n_heads=4, n_kv_heads=1, n_layers=2, d_ff=128, max_seq_len=16,
                 num_mixtures=4, image_ar_dim=8, patch_size=4, input_size=(16, 16), use_boi=True, repeats=1):
    from src.jetformer import JetFormer
    m = JetFormer(
        vocab_size=128,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        num_mixtures=num_mixtures,
        patch_size=patch_size,
        input_size=input_size,
        image_ar_dim=image_ar_dim,
        use_boi_token=use_boi,
        bos_id=1,
        boi_id=(2 if use_boi else None),
        nolabel_id=0,
        num_vocab_repeats=repeats,
        grad_checkpoint_transformer=False,
        flow_grad_checkpoint=False,
    )
    m.eval()
    return m


@torch.no_grad()
def test_compute_image_hidden_slicing_num_repeats_and_boi():
    torch.manual_seed(0)
    B = 2
    L_txt = 3
    L_img = 4
    for use_boi in (True, False):
        for repeats in (1, 2):
            model = _build_model(use_boi=use_boi, repeats=repeats)
            text_tokens = torch.randint(3, 50, (B, L_txt))
            image_tokens = torch.randn(B, L_img, model.image_ar_dim)
            text_mask = torch.ones(B, L_txt, dtype=torch.bool)
            text_first_mask = torch.tensor([True, False])

            # Direct forward to get full prelogits layout
            x, attn, pos = model.embed_sequence(text_tokens, image_tokens, text_first_mask, text_mask, None, None)
            if isinstance(model.transformer, torch.nn.ModuleList):
                h = x
                for layer in model.transformer:
                    h = layer(h, attn, pos, False)
                if not model.per_modality_final_norm:
                    h = model.final_norm(h)
            else:
                h = model.transformer(x, attn, pos)

            # Expected image regions
            repeats_total = repeats
            if use_boi:
                a_img = h[:, repeats_total * L_txt + 1 : repeats_total * L_txt + 1 + L_img]
                b_img = h[:, :L_img]
            else:
                a_img = h[:, repeats_total * L_txt : repeats_total * L_txt + L_img]
                b_img = h[:, :L_img]
            expected = torch.where(text_first_mask.view(B, 1, 1).expand(B, L_img, h.size(-1)), a_img, b_img)

            ih = model.compute_image_hidden(text_tokens, image_tokens, text_first_mask, text_mask)
            assert ih.shape == expected.shape
            torch.testing.assert_close(ih, expected, atol=1e-5, rtol=1e-5)


