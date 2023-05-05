# Monopoly AI Progress 4/14
### Reinforcement Learning development for Monopoly

Ciao y'all! Over the last semester, I've been developing a reinforcement learning agent for the boardgame Monopoly. Here's a quick glance at the progress:

The environment is already finished, where you can have any size Monopoly board and modular game rules. For the purposes of this project, I'm limiting the rules to just *selling* and *buying*, because other features like trading and auctioning are a bit more convoluted and require asynchronous decision making.

The environment is written in **Rust** using **bevy**. It's a kahoot-style setup where one process is the game board and every other process is some sort of player. The repository has the players split into two, where one is a real human player and the other is the reinforcement learning agent.

Networking is done via **naia**. This allows for the game board to process information asynchronously, not having to rely on when a player ends their turn. For our situation, naia handles a few messages:

- Forfeit, unavailable to the agent but useful for human players
- AlterOwnable, upgrading a property (unnecessary for now)
- SellOwnable, selling a property
- BuyOwnable, buying the property a player is standing on
- EndTurn, signal for the server to listen for the next player

And for the server-to-client side of things:

- BeginTurn, provides us with the action space (which will be an issue later)
- SendPlayer, synchronizing data between the server and client
- StartGame, signal to let clients provide interface, useful for human players

From there, the server listens for changes from the clients which dictates board changes.

The agent is where things get tricky. Notice how I said the action space would be an issue? The main issue with creating an agent for this is monopoly only lets you sell properties you own (makes sense, right?). But for the agent, we can't change the action space or else its policy gets thrown a bit out of wack.

I've been reading through a *[Catan agent article](https://settlers-rl.github.io)* for some guidance, and it seems that they are using *action space masking* to get around this. An array of legal actions will be provided to mask the probabilities of the agent's actions, providing the probabilities of only the valid actions. The same article also provides different ways to handle multi-decision actions (like trading or playing certain cards), which could be applicable to more of the Monopoly issues in the future.

That's it for now. The next steps is probably going to be implementing the algorithm with action space masking. I'm going to train the model on the standard 4 corners, 40 squares board that Monopoly primarily uses--since I'm likely to just mask all 40 squares as sellable possibilities at first. However, taking points from the article, it could be possible to make it handle dynamic board sizes.