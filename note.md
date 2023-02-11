## Original Implementation

- Input Size: 112
- Batch Size: 64
- Memory: 6350 MB
- 1 Epoch:
  - Time/Step : 0.11
  - Loss start: 1.5708
  - Loss end  : 0.4631

```
Model = MaskedAutoencoderViT(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
    (norm): Identity()
  )
  (blocks): ModuleList(
    (0-23): 24 x Block(
      (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=1024, out_features=3072, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=1024, out_features=1024, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (decoder_embed): Linear(in_features=1024, out_features=512, bias=True)
  (decoder_blocks): ModuleList(
    (0-7): 8 x Block(
      (norm1): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
      (attn): Attention(
        (qkv): Linear(in_features=512, out_features=1536, bias=True)
        (attn_drop): Dropout(p=0.0, inplace=False)
        (proj): Linear(in_features=512, out_features=512, bias=True)
        (proj_drop): Dropout(p=0.0, inplace=False)
      )
      (drop_path): Identity()
      (norm2): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
      (mlp): Mlp(
        (fc1): Linear(in_features=512, out_features=2048, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=2048, out_features=512, bias=True)
        (drop): Dropout(p=0.0, inplace=False)
      )
    )
  )
  (decoder_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  (decoder_pred): Linear(in_features=512, out_features=768, bias=True)
)
```

## Xformer Implementation

- Input Size: 112
- Batch Size: 64
- Memory: 8308 MB
- 1 Epoch:
  - Time/Epoch: 10.5 min
  - Time/Step : 0.17 sec
  - Loss start: 1.2193
  - Loss end  : 0.4631

## Best Attentions

- fourier_mix
  - `time: 0.1458  data: 0.0001  max mem: 3893`
- nystrom
  - `time: 0.2012  data: 0.0001  max mem: 5148`
- scaled_dot_product
  - `time: 0.2298  data: 0.0001  max mem: 5148`
- linformer
  - `time: 0.2279  data: 0.0001  max mem: 5205`
- orthoformer
  - `time: 0.8514  data: 0.0001  max mem: 5163`

```
Model = MaskedAutoencoderViT(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 1024, kernel_size=(16, 16), stride=(16, 16))
    (norm): Identity()
  )
  (encoder_blocks): ModuleList(
    (0-23): 24 x xFormerEncoderBlock(
      (wrap_att): Residual(
        (layer): PreNorm(
          (norm): FusedLayerNorm()
          (sublayer): MultiHeadDispatch(
            (attention): LinformerAttention(
              (E): Linear(in_features=50, out_features=12, bias=False)
              (F): Linear(in_features=50, out_features=12, bias=False)
              (attn_drop): Dropout(p=0.0, inplace=False)
            )
            (in_proj_container): InputProjection(
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (resid_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
        )
      )
      (wrap_ff): PostNorm(
        (norm): FusedLayerNorm()
        (sublayer): Residual(
          (layer): PreNorm(
            (norm): FusedLayerNorm()
            (sublayer): FusedMLP(
              (mlp): Sequential(
                (0): Linear(in_features=1024, out_features=4096, bias=False)
                (1): FusedDropoutBias(
                  (activation_pytorch): GELU(approximate='none')
                )
                (2): Linear(in_features=4096, out_features=1024, bias=False)
                (3): FusedDropoutBias(
                  (activation_pytorch): Identity()
                )
              )
            )
          )
        )
      )
    )
  )
  (encoder_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (decoder_embed): Linear(in_features=1024, out_features=512, bias=True)
  (decoder_blocks): ModuleList(
    (0-7): 8 x xFormerEncoderBlock(
      (wrap_att): Residual(
        (layer): PreNorm(
          (norm): FusedLayerNorm()
          (sublayer): MultiHeadDispatch(
            (attention): LinformerAttention(
              (E): Linear(in_features=50, out_features=12, bias=False)
              (F): Linear(in_features=50, out_features=12, bias=False)
              (attn_drop): Dropout(p=0.0, inplace=False)
            )
            (in_proj_container): InputProjection(
              (q_proj): Linear(in_features=512, out_features=512, bias=True)
              (k_proj): Linear(in_features=512, out_features=512, bias=True)
              (v_proj): Linear(in_features=512, out_features=512, bias=True)
            )
            (resid_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=512, out_features=512, bias=True)
          )
        )
      )
      (wrap_ff): PostNorm(
        (norm): FusedLayerNorm()
        (sublayer): Residual(
          (layer): PreNorm(
            (norm): FusedLayerNorm()
            (sublayer): FusedMLP(
              (mlp): Sequential(
                (0): Linear(in_features=512, out_features=2048, bias=False)
                (1): FusedDropoutBias(
                  (activation_pytorch): GELU(approximate='none')
                )
                (2): Linear(in_features=2048, out_features=512, bias=False)
                (3): FusedDropoutBias(
                  (activation_pytorch): Identity()
                )
              )
            )
          )
        )
      )
    )
  )
  (decoder_norm): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
  (decoder_pred): Linear(in_features=512, out_features=768, bias=True)
)
```
