#include <stdio.h>
#include <stdlib.h>

#include "eval_hand.h"

#define NUM_RANKS 13
#define NUM_SUITS 4
#define NUM_CARDS 52
#define COMBO_SIZE 7

uint64_t nChooseK(int n, int k) {
    if (k > n) return 0;
    uint64_t res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

void generate7CardCombinations(const char *output_file) {
    FILE *f = fopen(output_file, "wb");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    uint64_t total = 0;
    uint64_t totalCombos = nChooseK(NUM_CARDS, COMBO_SIZE);
    printf("Total combinations: %llu\n", (unsigned long long)totalCombos);

    for (int a = 0; a < NUM_CARDS - 6; ++a)
    for (int b = a + 1; b < NUM_CARDS - 5; ++b)
    for (int c = b + 1; c < NUM_CARDS - 4; ++c)
    for (int d = c + 1; d < NUM_CARDS - 3; ++d)
    for (int e = d + 1; e < NUM_CARDS - 2; ++e)
    for (int f_ = e + 1; f_ < NUM_CARDS - 1; ++f_)
    for (int g = f_ + 1; g < NUM_CARDS; ++g) {
        Cardset cs = emptyCardset();
        int cards[7] = {a, b, c, d, e, f_, g};
        for (int i = 0; i < 7; ++i) {
            int card = cards[i];
            int suit = card / NUM_RANKS;
            int rank = card % NUM_RANKS;
            addCardToCardset(&cs, suit, rank);
        }

        int val = rankCardset(cs);
        fwrite(&val, sizeof(int), 1, f);
        if (++total % 1000000 == 0) {
            printf("Processed %llu / %llu\n", (unsigned long long)total, (unsigned long long)totalCombos);
        }
    }

    fclose(f);
}

int main() {
    generate7CardCombinations("hand_ranks.bin");
    return 0;
}