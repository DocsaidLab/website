---
sidebar_position: 7
---

# 結果と議論

以前の実験を総合すると、良い結果を得られるモデルが構築できました。

ここでは、トレーニング過程でのいくつかの心得や経験について議論します。

---

- 私たちのモデルは SoTA に近いスコアを達成していますが、現実のシーンはこのデータセットよりもはるかに複雑です。そのため、このスコアに過度にこだわる必要はなく、私たちは単にモデルの有効性を証明したいと考えています。

- 実験の中で、現在設計されているモデルアーキテクチャは Zero-shot 能力があまり良くないことがわかりました。つまり、モデルは新しいシーンに対しては最適な結果を得るために微調整が必要です。今後は、より一般化能力の高いモデルアーキテクチャを探る必要があります。

- モデル設計の章でも述べたように、拡大誤差の課題を直接的に解決することはできません。そのため、「ヒートマップ回帰モデル」の安定性は「ポイント回帰モデル」よりもはるかに高いです。

- 私たちは、`FastViT_SA24` をヒートマップモデルのバックボーンとして使用しています。その効果と計算量が非常に良いためです。

- 実験を通じて、`BiFPN`（3 層）の方が `FPN`（6 層）よりも効果的であることがわかりました。そのため、`BiFPN` をネック部分の構成として使用することをお勧めします。ただし、私たちが実装した `BiFPN` では `einsum` 操作を使用しており、他の推論フレームワークで問題を引き起こす可能性があります。そのため、`BiFPN` を使用する際に変換エラーが発生した場合は、`FPN` モデルに変更することを検討してください。

- 「ヒートマップ回帰モデル」は安定した性能を発揮しますが、高解像度の特徴マップで監視を行う必要があるため、計算量は「ポイント回帰モデル」よりもはるかに大きくなります。

- それでも「ポイント回帰モデル」の利点を捨てることはできません。例えば、画面範囲外の角点を予測できること、計算量が少なく、後処理が簡単で迅速であることなどです。そのため、私たちは「ポイント回帰モデル」の効果を向上させるための探索と最適化を続けていきます。