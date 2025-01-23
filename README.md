
# BuckShotRouletteDQN v0.3.11

#
inputs: [(lives/4), (blanks/4), (round/3), [for (dogitem/6)+mask in dogitems], [for (dealeritem/6)+mask in dealeritems], (doghp/4), (dealer hp/4), (current shell/8)]
outputs: [item actions, shoot who = end token]
1: use beer etc. 0: shoot ai(self), 7 shoot dealer(opp)
(1+1+1+(8+8)+(8+8)+1+1+1), (6+2)
<div align="center">
  NLSCDDDQN(inputs, outputs, [80, 80, 80], skip_connections=[(0,2), (1,3)], use_noisy=True)
  <img src="https://github.com/user-attachments/assets/fb7969db-83fb-4e6e-bc88-7bb1509b464f"



</div>
