### Corpus de entrevistas: distribución de los campos

| clip   |   anios_experiencia | ultimo_puesto                    |   n_tecnologias | pretension   | disponibilidad             | nivel_ingles   |
|:-------|--------------------:|:---------------------------------|----------------:|:-------------|:---------------------------|:---------------|
| E01    |               4.500 | analista de datos senior         |               4 | 6500 PEN     | 2026-11-13 (relativa_dias) | intermedio     |
| E02    |               7.000 | backend developer                |               5 | 5000 PEN     | 2026-10-30 (inmediata)     | avanzado       |
| E03    |              10.000 | líder de proyectos de tecnología |               4 | 5200 PEN     | 2026-12-01 (fecha)         | basico         |
| E04    |               2.500 | ML engineer                      |               5 | 2800 USD     | 2026-11-29 (relativa_dias) | intermedio     |
| E05    |               6.000 | ingeniera de datos               |               6 | 9000 PEN     | 2026-11-06 (relativa_dias) | avanzado       |

### Corpus de llamadas: distribución y trampas

| clip   |   n_items |   unidades | promocion                       | metodo_pago   |   total_dicho |   total_correcto | autocorreccion   | total_erroneo   | tipo_error            |
|:-------|----------:|-----------:|:--------------------------------|:--------------|--------------:|-----------------:|:-----------------|:----------------|:----------------------|
| V01    |         4 |          4 | combo familiar                  | yape          |            87 |               87 | True             | False           | nan                   |
| V02    |         4 |          5 | martes de pizza 2x1 en medianas | efectivo      |           102 |               70 | True             | True            | promocion_no_aplicada |
| V03    |         3 |          4 | código BARRIO15                 | tarjeta       |            85 |               95 | False            | True            | error_suma            |
| V04    |         4 |          5 | nan                             | plin          |            66 |               66 | True             | False           | nan                   |
| V05    |         3 |          3 | nan                             | transferencia |            63 |               74 | True             | True            | total_pedido_previo   |

### Benchmark GPU (Tesla T4, 15360 MiB, 580.82.07)

| modelo         | dispositivo_pedido   | dispositivo_real   | precision   | modo_glosario   |   wer |   cer |   seg_por_min_audio |   vram_pico_mb |   vram_modelo_mb |
|:---------------|:---------------------|:-------------------|:------------|:----------------|------:|------:|--------------------:|---------------:|-----------------:|
| large-v3-turbo | cpu                  | cpu                | int8        | ninguno         | 0.040 | 0.018 |              70.130 |        556.188 |            0.000 |
| large-v3-turbo | cuda                 | cuda               | float16     | hotwords        | 0.250 | 0.207 |               2.758 |       2524.588 |         1968.400 |
| large-v3-turbo | cuda                 | cuda               | float16     | initial_prompt  | 0.048 | 0.020 |               1.978 |       2390.188 |         1834.000 |
| large-v3-turbo | cuda                 | cuda               | float16     | ninguno         | 0.045 | 0.019 |               1.854 |       2390.188 |         1834.000 |
| large-v3-turbo | cuda                 | cuda               | int8        | ninguno         | 0.052 | 0.022 |               1.808 |       1590.188 |         1034.000 |
| small          | cuda                 | cuda               | float16     | ninguno         | 0.077 | 0.031 |               1.694 |       1327.788 |          877.600 |
| small          | cuda                 | cuda               | int8        | ninguno         | 0.079 | 0.035 |               1.706 |       1046.188 |          490.000 |

### ¿Cambia el texto entre float16 e int8?

| modelo         | clip   | texto_identico   |   wer_entre_float16_e_int8 |
|:---------------|:-------|:-----------------|---------------------------:|
| small          | E01    | False            |                      0.009 |
| small          | E02    | False            |                      0.046 |
| small          | E03    | False            |                      0.021 |
| small          | E04    | False            |                      0.025 |
| small          | E05    | False            |                      0.015 |
| large-v3-turbo | E01    | False            |                      0.005 |
| large-v3-turbo | E02    | True             |                      0.000 |
| large-v3-turbo | E03    | False            |                      0.012 |
| large-v3-turbo | E04    | False            |                      0.069 |
| large-v3-turbo | E05    | False            |                      0.012 |

### ¿Cambia el texto entre GPU y CPU (int8)?

| clip   | texto_identico   |   wer_entre_si |
|:-------|:-----------------|---------------:|
| E01    | True             |          0.000 |
| E02    | False            |          0.008 |

### Tarea 1 — capa transcripción (caché local)

| clip   |   wer |   cer | dispositivo   |   seg_por_min_audio |
|:-------|------:|------:|:--------------|--------------------:|
| E01    | 0.041 | 0.021 | cpu           |              26.640 |
| E02    | 0.038 | 0.014 | cpu           |              28.900 |
| E03    | 0.046 | 0.018 | cpu           |              49.340 |
| E04    | 0.055 | 0.025 | cpu           |              31.580 |
| E05    | 0.055 | 0.025 | cpu           |              34.720 |

### Tarea 1 — capa registro: exactitud por campo (n = 5, IC de Wilson)

| campo               | llm_ollama        | llm_openai        | reglas            |
|:--------------------|:------------------|:------------------|:------------------|
| anios_experiencia   | 1.00 [0.57, 1.00] | 1.00 [0.57, 1.00] | 1.00 [0.57, 1.00] |
| disponibilidad      | 0.20 [0.04, 0.62] | 1.00 [0.57, 1.00] | 1.00 [0.57, 1.00] |
| nivel_ingles        | 0.80 [0.38, 0.96] | 1.00 [0.57, 1.00] | 1.00 [0.57, 1.00] |
| pretension_salarial | 0.60 [0.23, 0.88] | 1.00 [0.57, 1.00] | 1.00 [0.57, 1.00] |
| registro_completo   | 0.00 [0.00, 0.43] | 0.40 [0.12, 0.77] | 0.40 [0.12, 0.77] |
| tecnologias         | 0.40 [0.12, 0.77] | 0.40 [0.12, 0.77] | 0.40 [0.12, 0.77] |
| ultimo_puesto       | 0.40 [0.12, 0.77] | 1.00 [0.57, 1.00] | 0.80 [0.38, 0.96] |

F1 macro de `nivel_ingles`: llm_ollama = 0.889, llm_openai = 1.000, reglas = 1.000

### Tarea 1 — error absoluto medio

| metodo     |   error_abs_anios |   error_abs_salario |
|:-----------|------------------:|--------------------:|
| llm_ollama |             0.000 |             999.000 |
| llm_openai |             0.000 |               0.000 |
| reglas     |             0.000 |               0.000 |

### Verificación literal contra corrección

| metodo     | verificacion   |   False |   True |
|:-----------|:---------------|--------:|-------:|
| llm_ollama | sin_verificar  |   5.000 |  0.000 |
| llm_ollama | verificado     |   8.000 | 17.000 |
| llm_openai | verificado     |   3.000 | 27.000 |
| reglas     | vacio          |   1.000 |  0.000 |
| reglas     | verificado     |   3.000 | 26.000 |

### Glosario (caché local, int8)

| modo           |   wer_sin |   wer_con |   dif_wer_media | ic95_dif         |   terminos_total |   recuperados_sin |   recuperados_con |
|:---------------|----------:|----------:|----------------:|:-----------------|-----------------:|------------------:|------------------:|
| hotwords       |     0.047 |     0.272 |           0.225 | [+0.166, +0.267] |               27 |                21 |                22 |
| initial_prompt |     0.047 |     0.050 |           0.003 | [-0.001, +0.009] |               27 |                21 |                22 |

### Tarea 2 — escalera de degradación (ambas capas)

| nivel   | metodo     |   wer_medio |   cer_medio |   exactitud_campos |   exactitud_items |   exactitud_total_dicho |
|:--------|:-----------|------------:|------------:|-------------------:|------------------:|------------------------:|
| L0      | llm_ollama |       0.044 |       0.016 |              0.800 |             0.600 |                   1.000 |
| L0      | llm_openai |       0.044 |       0.016 |              0.960 |             1.000 |                   1.000 |
| L0      | reglas     |       0.044 |       0.016 |              0.680 |             0.600 |                   0.800 |
| L1      | llm_ollama |       0.071 |       0.032 |              0.720 |             0.200 |                   0.800 |
| L1      | llm_openai |       0.071 |       0.032 |              0.920 |             0.800 |                   1.000 |
| L1      | reglas     |       0.071 |       0.032 |              0.640 |             0.600 |                   0.800 |
| L2      | llm_ollama |       0.154 |       0.116 |              0.680 |             0.000 |                   1.000 |
| L2      | llm_openai |       0.154 |       0.116 |              0.800 |             0.400 |                   1.000 |
| L2      | reglas     |       0.154 |       0.116 |              0.680 |             0.400 |                   0.800 |
| L3      | llm_ollama |       0.423 |       0.277 |              0.560 |             0.000 |                   1.000 |
| L3      | llm_openai |       0.423 |       0.277 |              0.680 |             0.200 |                   1.000 |
| L3      | reglas     |       0.423 |       0.277 |              0.480 |             0.000 |                   0.800 |

### Tarea 2 — validación aritmética: detectados vs omitidos

| nivel   | metodo     |   trampas |   detectados |   omitidos |   falsas_alarmas |   no_verificables |   diagnosticos_evaluados |   diagnosticos_correctos |
|:--------|:-----------|----------:|-------------:|-----------:|-----------------:|------------------:|-------------------------:|-------------------------:|
| L0      | llm_ollama |         3 |            2 |          1 |                1 |                 0 |                        3 |                        1 |
| L0      | llm_openai |         3 |            3 |          0 |                0 |                 0 |                        3 |                        3 |
| L0      | reglas     |         3 |            2 |          1 |                0 |                 2 |                        2 |                        2 |
| L1      | llm_ollama |         3 |            2 |          1 |                1 |                 1 |                        3 |                        2 |
| L1      | llm_openai |         3 |            3 |          0 |                0 |                 0 |                        3 |                        3 |
| L1      | reglas     |         3 |            3 |          0 |                0 |                 1 |                        3 |                        3 |
| L2      | llm_ollama |         3 |            3 |          0 |                2 |                 0 |                        5 |                        1 |
| L2      | llm_openai |         3 |            3 |          0 |                0 |                 0 |                        3 |                        3 |
| L2      | reglas     |         3 |            3 |          0 |                0 |                 1 |                        3 |                        3 |
| L3      | llm_ollama |         3 |            3 |          0 |                1 |                 0 |                        4 |                        2 |
| L3      | llm_openai |         3 |            2 |          1 |                1 |                 0 |                        3 |                        3 |
| L3      | reglas     |         3 |            3 |          0 |                1 |                 1 |                        4 |                        3 |

### SNR medida en L3

| nivel   | clip   |   snr_objetivo_db |   snr_medida_voz_activa_db |   snr_medida_global_db |
|:--------|:-------|------------------:|---------------------------:|-----------------------:|
| L3      | V01    |             5.000 |                      5.000 |                  3.460 |
| L3      | V02    |             5.000 |                      5.000 |                  3.460 |
| L3      | V03    |             5.000 |                      5.000 |                  3.460 |
| L3      | V04    |             5.000 |                      5.000 |                  3.460 |
| L3      | V05    |             5.000 |                      5.000 |                  3.470 |

### WER contra exactitud del registro (todas las combinaciones)

| tarea   | nivel   | clip   | metodo     |   wer |   exactitud_registro |
|:--------|:--------|:-------|:-----------|------:|---------------------:|
| llamada | L3      | V05    | llm_ollama | 0.644 |                0.600 |
| llamada | L3      | V05    | llm_openai | 0.644 |                0.600 |
| llamada | L3      | V05    | reglas     | 0.644 |                0.600 |
| llamada | L3      | V04    | llm_ollama | 0.538 |                0.600 |
| llamada | L3      | V04    | llm_openai | 0.538 |                0.800 |
| llamada | L3      | V04    | reglas     | 0.538 |                0.200 |
| llamada | L3      | V01    | reglas     | 0.360 |                0.400 |
| llamada | L3      | V01    | llm_openai | 0.360 |                0.600 |

| tarea   | nivel   | clip   | metodo     |   wer |   exactitud_registro |
|:--------|:--------|:-------|:-----------|------:|---------------------:|
| llamada | L1      | V02    | llm_ollama | 0.017 |                0.600 |
| llamada | L1      | V02    | reglas     | 0.017 |                0.600 |
| llamada | L1      | V02    | llm_openai | 0.017 |                0.800 |
| llamada | L0      | V02    | llm_ollama | 0.017 |                0.800 |
| llamada | L0      | V02    | llm_openai | 0.017 |                0.800 |
| llamada | L0      | V02    | reglas     | 0.017 |                0.600 |
| llamada | L2      | V01    | llm_ollama | 0.030 |                0.800 |
| llamada | L2      | V01    | llm_openai | 0.030 |                1.000 |
