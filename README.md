
# BuckShotRouletteDQN v0.4.2

#
inputs: [(lives/4), (blanks/4), (shell), [item/6 for item in AI_items], [item/6 for item in DEALER_items], (AI hp/4), (dealer hp/4), (current shell/8)]
outputs: [item actions, shoot who = end token]
1: use beer etc. 0: shoot ai(self), 7 shoot dealer(opp)
(1+1+1+(8)+(8)+1+1+1), (6+2)
<div align="center">
  NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,2), (1,3)], use_noisy=True)
  <img src="https://github.com/user-attachments/assets/7b3b949c-3da5-44ad-a193-7755f72e7b79"
</div>
